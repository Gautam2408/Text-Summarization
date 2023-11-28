from django.shortcuts import render, redirect
from requests import request

# Create your views here.
# Dependencies use for TF-IDF and Transformer
import tensorflow as tf
from tensorflow import keras
from logging import PercentStyle
import math
import string
import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer,punkt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import gc
import requests
from bs4 import BeautifulSoup
from django.core.files.storage import FileSystemStorage
import fitz  # PyMuPDF


#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

def redirect_root(request):
    return redirect('/index')
    
def index(request):
    return render(request, 'index.html')

def summary_url(request):
    return render(request, 'url.html')

def summary_doc(request):
    return render(request, 'doc.html')

def summary_text(request):
    return render(request, 'text.html')

def contact(request):
    return render(request, 'index.html')

def summarized_text(request):
    if request.method == "POST":
        text = request.POST.get("textsum")
        volume = request.POST.get("vol")
        text_sum = generate_summary_for_text(text,volume)
        context = { "text_sum" : text_sum ,
                   "text" : text}
        return render(request,'gen_text.html',context)

def summarized_url(request):
    if request.method == "POST":
        text = request.POST.get("urlinput")
        volume = request.POST.get("vol")
        #url_sum = generate_summary_for_url(text,volume)

        # Retrieve page text
        page = requests.get(text).text

        # Turn page into BeautifulSoup object to access HTML tags
        soup = BeautifulSoup(page)

        # Get text from all <p> tags.
        p_tags = soup.find_all('p')

        # Get the text from each of the “p” tags and strip surrounding whitespace.
        p_tags_text = [tag.get_text().strip() for tag in p_tags]

        # Filter out sentences that contain newline characters '\n' or don't contain periods.
        sentence_list = [sentence for sentence in p_tags_text if not '\n' in sentence]
        sentence_list = [sentence for sentence in sentence_list if '.' in sentence]

        # Combine list items into string.
        article = ' '.join(sentence_list)
        textsum = generate_summary_for_text(article,volume)
        context = { "text" : article ,
                     "text_sum" : textsum}
        return render(request,'gen_text.html',context)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    doc.close()
    return text

def summarized_doc(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('pdfFile')
        volume = request.POST.get("vol")

        if uploaded_file:
            # Save the uploaded file
            fs = FileSystemStorage()
            pdf_path = fs.save(uploaded_file.name, uploaded_file)

            # Extract text from the uploaded PDF
            extracted_text = extract_text(pdf_path)
            
            text_sum = generate_summary_for_doc(extracted_text,volume)

            context = {
            "text" : extracted_text,
            "text_sum" : text_sum
            }
            return render(request, 'gen_text.html', context)


# Length adjuster

def _len_convter(context, con_length):
    choice = int(con_length)
    if (choice == 2):
        max_le = int(len(context) * 0.5)
        min_le = int(len(context) * 0.1)
    elif (choice == 1):
        max_le = int(len(context) * 0.3)
        min_le = int(len(context) * 0.1)
    elif (choice == 0):
        max_le = int(len(context) * 0.15)
        min_le = int(len(context) * 0.1)
    return max_le, min_le


# Create Frequency Table
def _create_frequency_table(text_string) -> dict:
    '''
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    Lemmatizer - an algorithm that brings words to its root word
    :return type: dict
    '''
    stopWords = set(stopwords.words("english"))
    punkt = set()
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    freqTable = dict()
    for word in words:
        # word = ps.stem(word)
        word = lemmatizer.lemmatize(word)
        if (word in stopWords or word in string.punctuation):
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


# Create Frequency Matrix
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    # ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            # word = ps.stem(word)
            word = lemmatizer.lemmatize(word)
            if (word in stopWords or word in string.punctuation):
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


# Create Document Per Words
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


# Create TF Matrix
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


# Create IDF Matrix
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))  # log10

        idf_matrix[sent] = idf_table

    return idf_matrix


# Create TF-IDF Matrix
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


# Create Score Sentences
def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


# Find Average Score
def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


# Run Extractive Summariztion for Text Using TF-IDF
def run_summarization(text):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    # print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    # print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    # print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    # print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    # print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    # print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    # print(sentence_scores)

    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    # print(threshold)

    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 0.8 * threshold)
    return summary


def call_rouge(article, exc_res, abs_res):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores1 = scorer.score(article, exc_res)
    scores2 = scorer.score(exc_res, abs_res)
    print("Rouge Score for Article vs Extracted Summary : ")
    print(scores1)
    print()
    print()
    print("Rouge Score for Extracted Summary vs Abstracted Summary")
    print(scores2)


# Model for Document Summary.
def abs_summary_for_doc(article,_len):

  _model = AutoModelForSeq2SeqLM.from_pretrained(
                "philschmid/bart-large-cnn-samsum"
            )

  _tokenizer = AutoTokenizer.from_pretrained(
                "philschmid/bart-large-cnn-samsum"
            )

                                           

  summarizer = pipeline(
                      "summarization", 
                      model=_model, 
                      tokenizer=_tokenizer,
                      device=0
                  )

  result = summarizer(
            article,
            min_length=_len[1], 
            max_length=_len[0],
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True
    )

  del _model
  del _tokenizer
  del summarizer
  gc.collect()
  return result

    

def generate_summary_for_doc(text, con_length):
    # print(tf.config.list_physical_devices('GPU'))
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print(torch.version.cuda)

    article = text.strip().replace("\n", "")
    result = run_summarization(article)
    _length = _len_convter(result, con_length)
    abs_result = abs_summary_for_doc(result, _length)

    call_rouge(article, result, abs_result[0]['summary_text'])

    return abs_result[0]['summary_text']


#Model for URL summary
def abs_summary_for_url(article,_len):

  tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

  model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum")
                                           

  summarizer = pipeline(
                      "summarization", 
                      model=model, 
                      tokenizer=tokenizer,
                      device=0
                  )

  result = summarizer(
            article,
            min_length=_len[1], 
            max_length=_len[0],
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True
    )

    
  del tokenizer
  del model
  del summarizer
  gc.collect()  
  return result

def generate_summary_for_url(text, con_length):
    # print(tf.config.list_physical_devices('GPU'))
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print(torch.version.cuda)

    article = text.strip().replace("\n", "")
    result = run_summarization(article)
    _length = _len_convter(result, con_length)
    abs_result = abs_summary_for_url(result, _length)

    call_rouge(article, result, abs_result[0]['summary_text'])

    return abs_result[0]['summary_text']

# Model for text
def abs_summary_for_text(article,_len):

  tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

  model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum")
                                           

  summarizer = pipeline(
                      "summarization", 
                      model=model, 
                      tokenizer=tokenizer,
                      device=0
                  )

  result = summarizer(
            article,
            min_length=_len[1], 
            max_length=_len[0],
            no_repeat_ngram_size=3, 
            encoder_no_repeat_ngram_size =3,
            clean_up_tokenization_spaces=True,
            repetition_penalty=3.7,
            num_beams=4,
            early_stopping=True
    )
    
  del tokenizer
  del model
  del summarizer
  gc.collect()
  return result

def generate_summary_for_text(text, con_length):
    # print(tf.config.list_physical_devices('GPU'))
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print(torch.version.cuda)

    article = text.strip().replace("\n", "")
    result = run_summarization(article)
    _length = _len_convter(result, con_length)
    abs_result = abs_summary_for_text(result, _length)
    call_rouge(article, result, abs_result[0]['summary_text'])
    return abs_result[0]['summary_text']

# Run Abstractive Summariztion For Text Using Bart

#(Code for common model for 3 pages commented below from line 256-267)

# def abs_summary(summary, _len):
#     summarizer = pipeline("summarization")
#     summarized = summarizer(summary, min_length=_len[1], max_length=_len[0])
#     return summarized

# def generate_summary(text, con_length):
#     article = text.strip().replace("\n", "")
#     result = run_summarization(article)
#     _length = _len_convter(result, con_length)
#     abs_result = abs_summary(result, _length)
#     call_rouge(article, result, abs_result[0]['summary_text'])
#     return abs_result[0]['summary_text']
    

# if __name__ == '__main__':
#     # Input from user
#     article_raw = '''Data mining is the process of extracting and discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems.Data mining is an interdisciplinary subfield of computer science and statistics with an overall goal of extracting information (with intelligent methods) from a data set and transforming the information into a comprehensible structure for further use.Data mining is the analysis step of the "knowledge discovery in databases" process, or Aside from the raw analysis step, it also involves database and data management aspects, data pre-processing, model and inference considerations, interestingness metrics, complexity considerations, post-processing of discovered structures, visualization, and online updating.
# The term "data mining" is a misnomer because the goal is the extraction of patterns and knowledge from large amounts of data, not the extraction (mining) of data itself.It also is a buzzword and is frequently applied to any form of large-scale data or information processing (collection, extraction, warehousing, analysis, and statistics) as well as any application of computer decision support system, including artificial intelligence (e.g., machine learning) and business intelligence. The book Data mining: Practical machine learning tools and techniques with Java (which covers mostly machine learning material) was originally to be named Practical machine learning, and the term data mining was only added for marketing reasons.Often the more general terms (large scale) data analysis and analytics—or, when referring to actual methods, artificial intelligence and machine learning—are more appropriate.
# The actual data mining task is the semi-automatic or automatic analysis of large quantities of data to extract previously unknown, interesting patterns such as groups of data records (cluster analysis), unusual records (anomaly detection), and dependencies (association rule mining, sequential pattern mining). This usually involves using database techniques such as spatial indices. These patterns can then be seen as a kind of summary of the input data, and may be used in further analysis or, for example, in machine learning and predictive analytics. For example, the data mining step might identify multiple groups in the data, which can then be used to obtain more accurate prediction results by a decision support system. Neither the data collection, data preparation, nor result interpretation and reporting is part of the data mining step, although they do belong to the overall KDD process as additional steps.
# The difference between data analysis and data mining is that data analysis is used to test models and hypotheses on the dataset, e.g., analyzing the effectiveness of a marketing campaign, regardless of the amount of data. In contrast, data mining uses machine learning and statistical models to uncover clandestine or hidden patterns in a large volume of data.
# The related terms data dredging, data fishing, and data snooping refer to the use of data mining methods to sample parts of a larger population data set that are (or may be) too small for reliable statistical inferences to be made about the validity of any patterns discovered. These methods can, however, be used in creating new hypotheses to test against the larger data populations.'''
#     article = article_raw.strip().replace("\n", "")

#     # print(article)
#     # Run Extractive Summerization for given article
#     #result = run_summarization(article)
#     #print("The summary after tfidf : \n"+result)
#     #print("The length of orginal article : "+str(len(article)))
#     #print("The length after tfidf : "+str(len(result)))
#     # _length = _len_convter(result,1)
#     # abs_result = abs_summary(result, _length)
#     abs_result = generate_summary_for_text(article, 2)    
#     print("The summary after all :\n"+abs_result)
#     print("The final summary length : "+str(len(abs_result)))
    #   req = requests.get("https://hackaday.com/2021/04/20/a-look-at-the-most-aerodynamic-cars-ever-built/")
    #   soup = BeautifulSoup(req.content, "html.parser")
    #   TEXT = soup.find('div', {'itemprop': 'articleBody'}). get_text()
    #   with open('article.pkl', 'wb') as f:
    #     pickle.dump(TEXT, f)
    # with open('article.pkl', 'rb') as f:
    #     TEXT = pickle.load(f)


