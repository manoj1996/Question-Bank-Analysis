from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import os
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.sparse


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

# TO DO: remove numbers and special characters, retain only words
def processText(text):
	text = text.replace('-',' ')
	return text

def findTfidfMatrix(filenames):
	filecontent = []

	for filename in filenames:
		f = open(filename,'r')
		filecontent.append(processText(f.read()))
		f.close()
		
	tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
	tfidf_matrix = tfidf_vectorizer.fit_transform(filecontent)
	
	return tfidf_matrix,(tfidf_vectorizer.get_feature_names())
	

# returns [ (cluster number,[topics...])... ]
def clusterTopics(tfidf_matrix,filenames,num_clusters=10):
	km = KMeans(n_clusters=num_clusters)
	km.fit(tfidf_matrix)
	clusters = km.labels_.tolist()
	topics = (list(zip(clusters,filenames)))
	#topics.sort(key=lambda x:(x[0],int(x[1].split('_')[1])))
	topics.sort()
	clusteredTopics = []
	clusterNum = -1
	for x in topics:
		if x[0]!=clusterNum:
			clusterNum += 1
			clusteredTopics.append((clusterNum,[]))
		clusteredTopics[clusterNum][1].append(x[1])
		
	return clusteredTopics

def plotGraph(tfidf_matrix):

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	

	cx = scipy.sparse.coo_matrix(matrix)
	x = []
	y = []
	z = []
	
	ax.scatter(cx.row,cx.col,cx.data, c='r', marker='o')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()


if __name__=='__main__':

	filenames = ['training/'+x for x in list(os.listdir('training/'))]
	matrix,words = findTfidfMatrix(filenames)
	
	topics = clusterTopics(matrix,filenames)
	for x in topics:
		print 'Cluster '+str(x[0])+':'
		for y in x[1]:
			print y
		print '-'*100
		
	plotGraph(matrix)



