# Karan and Manoj

import os
import nltk
import re
from xlrd import open_workbook
import shutil
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from cluster import *

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
    

# extract and write questions to text files
def extractQuestions(filename):
	wb = open_workbook(filename)
	# data read from excel sheet
	data = []
	
	for sheet in wb.sheets():
		number_of_rows = sheet.nrows
		number_of_columns = sheet.ncols
		
		for row in range(1, number_of_rows):
			values = []
			for col in range(number_of_columns):
				value  = sheet.cell(row,col).value
				values.append(value)
			
			data.append((values[0],values[1]))
	
	return data


# writes each question and marks obtained in it to a new file in "test" directory
# data - list of tuples - [(question,marks scored),...]
def writeQuestions(data):

	if os.path.exists("test"):
		shutil.rmtree('test')
	os.makedirs("test")

	qnum = 1
	basename = 'Question'
	extension = '.txt'
	#marksFile = open('test/marks.txt','w')
	marksScored = {}
	
	for x in data:
		question = x[0]
		marks = x[1]
		fname = basename + str(qnum) + extension
		f = open('test/'+fname,'w')
		f.write(question)
		f.close()
		#marksFile.write(str(qnum)+':'+str(marks)+'\n')
		qnum += 1
	#marksFile.close()


# returns frequency matrix, feature names(words), labels 
def findCountMatrix(filenames):
	filecontent = []
	labels = []

	for filename in filenames:
		f = open(filename,'r')
		sentences = nltk.sent_tokenize(processText(f.read()))
		labels.extend([filename]*len(sentences))
		filecontent.extend(sentences) 
		f.close()
		
		
	vectorizer = CountVectorizer(tokenizer=tokenize_and_stem, analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
	freq_matrix = vectorizer.fit_transform(filecontent)
	
	return freq_matrix,vectorizer.get_feature_names(),labels


# created once
stop_words = set(stopwords.words('english'))

# returns list of words after removing stop words
def removeStopWords(text):
	wordList = [x for x in nltk.word_tokenize(text) if x not in stop_words]
	return wordList  				

# returns frequency matrix of questions after removing stopwords and stemming
def getQuestionMatrix(matrix,labels,data):
	
	row = []
	col = []
	value = []
	qnum = 0
	m,n = matrix.shape
	matrix = matrix.toarray()
	
	for x in process_questions(data):
		#question = x[0]
		#words = removeStopWords(question)
		words = x
		for word in words:
			if word in labels:
				i = labels.index(word)
				if i not in col:
					col.append(i)
					value.append(1)
					row.append(qnum)
				else:
					j = col.index(i)
					value[j] += 1
		qnum += 1
	
	row = np.array(row)
	col = np.array(col)
	value = np.array(value)

	sparse_matrix = coo_matrix((value,(row,col)),shape=(qnum,n))
	return sparse_matrix
	

# tokenize questions, remove stop words, stem and return list of words - [[q1 words],[q2 words]...]	
def process_questions(data):
	res = []
	for question,score in data:
		stemmed_words = [stemmer.stem(x) for x in removeStopWords(question)]
		res.append(stemmed_words)
	return res


# data - list of tuples - [(question,marks scored),...]
# train classifier using tf matrix of pdf, classify questions 
def classify(data):
	filenames = list(os.listdir('training/'))
	training_filenames = ['training/'+x for x in filenames]
	frequency_matrix,word_list,labels = findCountMatrix(training_filenames)
	
	# test data set - list of stemmed words of questions
	test_dataset = getQuestionMatrix(frequency_matrix,word_list,data)
	
	clf = MultinomialNB()
	# matrix,labels passed
	clf.fit(frequency_matrix,labels)
	
	# get probability scores for each topic
	probability_matrix = clf.predict_proba(test_dataset)
	#list of classes in the order they are assigned by the classifier
	classes = clf.classes_

	# print question and topic to which they belong
	qnum = 0
	for x in probability_matrix:
		x = list(x)
		maxscr = max(x)
		index = []
		for i in range(len(x)):
			if x[i]==maxscr:
				index.append(i)
			
		print str(data[qnum][0]) + ':'
		chapters = [classes[n] for n in index]
		print '\n'.join(chapters)
		print 'score: ',str(maxscr)
		print '-'*50
		print
		qnum += 1	
	
	

if __name__=="__main__":
	data = extractQuestions("questionTrain.txt")
	#writeQuestions(data)
	#training_tfidf,test_tfidf = getTfidfScores()
	classify(data)

