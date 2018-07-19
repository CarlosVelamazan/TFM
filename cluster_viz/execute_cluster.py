# -*- coding: utf-8 -*-
"""
@author: Carlos Velamazán Mas
"""
import time
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.manifold import MDS

import matplotlib
import mpld3
# import matplotlib.pyplot
# from matplotlib import pyplot as plt



##########################
#GET TEXT FROM DATASETS  #
##########################
class Cluster():
    def __init__(self, title_list, text_list, num_clusters=2):
        self.NUM_CLUSTERS = num_clusters
#        self.PATH = './data/'
#        self.TITLES_NAME = 'titles.txt'
#        self.TEXTS_NAME = 'texts.txt'
        self.title_list = title_list
        self.text_list = text_list
        
    def get_data(self):
        
        clean_texts = []
        #Se importan los dos datasets
#        titles = open(path+titles_name).read().split('\nNUEVOTEXTO\n')
#        texts = open(path+texts_name).read().split('\nNUEVOTEXTO\n')
        
        titles = self.title_list
        texts = self.text_list
        
        #Se limitan a 100 filas
        titles = titles[:100]      
        texts = texts[:100]        
        
        #Se limpia el texto eliminando el formato html y convirtiendolo a unicode
        for text in texts:
            text = BeautifulSoup(text, 'html.parser').getText()
            clean_texts.append(text)        
        texts = clean_texts
        
        ranks = []
        for i in range(0,len(clean_texts)):
            ranks.append(i)
        
        return titles, texts, ranks
    
    def tokenize_and_stem(self, text):
        #Cargamos el stemmer
        stemmer = SnowballStemmer("english")
        
        # Tokenizamos primero las frases y luego las palabras 
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # Eliminamos numeros, puntuacion
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        
        return stems


    def tokenize_only(self, text):
        # Tokenizamos primero las frases y luego las palabras
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # Eliminamos numeros, puntuacion
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
                
        return filtered_tokens
    
    
    def tokenize_stem_texts(self, texts):
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in texts:
            allwords_stemmed = self.tokenize_and_stem(i)
            totalvocab_stemmed.extend(allwords_stemmed)
            
            allwords_tokenized = self.tokenize_only(i)
            totalvocab_tokenized.extend(allwords_tokenized)
        
        #Creamos un dataframe que actua como diccionario para obtener el token usando la palabra "stematizada"
        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    
        return vocab_frame
    
    def vectorize_texts_tfidf(self, texts):
        #Cargamos la lista de stop words de la libreria nltk
        stopwords = nltk.corpus.stopwords.words('english')
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=self.tokenize_and_stem, ngram_range=(1,3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        #guardamos los terminos de la matriz y calculamos las distancias
        terms = tfidf_vectorizer.get_feature_names()        
        dist = 1 - cosine_similarity(tfidf_matrix)
        
        return terms, dist, tfidf_matrix
        
    def apply_kmeans(self, num_clusters, tfidf_matrix):
        #Instanciamos el modelo kmeans
        km = KMeans(n_clusters=num_clusters)
        #Aplicamos el modelo a los datos
        km.fit(tfidf_matrix)
        #Obtenemos la lista de clústers
        clusters = km.labels_.tolist()
        
        return clusters, km
    
    def count_clusters(self, dataframe):
        #Contamos el número de documentos de cada cluster
        frame = dataframe
        num_docs_0 = frame['cluster'].value_counts()[0]
        num_docs_1 = frame['cluster'].value_counts()[1]
        
        return num_docs_0, num_docs_1
    
    def create_dataframe(self, titles, ranks, texts, clusters):
        #Creamos un dataframe a partir de los datos obtenidos anteriormente
        docs = { 'title': titles, 'rank': ranks, 'text': texts, 'cluster': clusters}
        frame = pd.DataFrame(docs, index = [clusters] , columns = ['rank', 'title', 'cluster'])
        
        return frame
    
    def get_top_terms_titles(self, num_clusters, km, vocab_frame, terms, frame):
    
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        cluster0_words = []
        cluster0_titles = []
        cluster1_words = []
        cluster1_titles = []
        for i in range(num_clusters):    
            if i == 0:
                #Add words and titles of the clusters
                for ind in order_centroids[i, :5]:
                    cluster0_words.append(vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])        
                for title in frame.loc[i]['title'].values.tolist()[:10]:
                    cluster0_titles.append(title)    
            elif i == 1:
                for ind in order_centroids[i, :5]:
                    cluster1_words.append(vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])
                for title in frame.loc[i]['title'].values.tolist()[:10]:
                    cluster1_titles.append(title)
        
        return cluster0_words, cluster0_titles, cluster1_words, cluster1_titles
    
    def reduce_dimensions(self, dist):
        #Reducimos la dimensionalidad del dataset para poder representarlo en el eje de coordenadas
        MDS()
        # n_components=2 porque vamos a representar los datos en dos dimensiones
        # "precomputed" porque proporcionamos una matriz de distancias
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)        
        pos = mds.fit_transform(dist)        
        xs, ys = pos[:, 0], pos[:, 1]
        
        return xs, ys
    
    
    def visualize_results(self, c0words, c1words, xs, ys, clusters, titles):
        
        #Creamos un diccionario con los colores a usar en la visualizacion
        cluster_colors = {0: '#1b9e77', 1: '#d95f02'}
        
        #Obtenemos las palabras clave que se usarán para la leyenda
        words0 = ''
        words1 = ''
        for i in range(0,len(c0words)):
            words0 = words0+c0words[i]+', '
            words1 = words1+c1words[i]+', '
        words0.strip(', ')
        words1.strip(', ')
        cluster_names = {0: words0, 1: words1}
        
        #Creamos un dataframe con los resultados de la 
        #reduccion de dimensionalidad, los clusters y los titulos
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 
        
        #Agrupamos por clusters
        groups = df.groupby('label')
        
        #Definimos un CSS para la visualización
        css = """
        text.mpld3-text, div.mpld3-tooltip {
          font-family:Arial, Helvetica, sans-serif;
        }
        
        g.mpld3-xaxis, g.mpld3-yaxis {
        display: none; }
        """
        
        # Plot 
        fig, ax = plt.pyplot.subplots(figsize=(14,6)) #set plot size
        ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling
        
        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
            ax.set_aspect('auto')
            labels = [i for i in group.title]
            
            #Configuramos el tooltip
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, \
                                               voffset=10, hoffset=10, css=css)
            #Conectamos el tooltip a la figura
            mpld3.plugins.connect(fig, tooltip, TopToolbar())    
            
            #Volvemos invisibles las marcas de los ejes
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            
            #Volvemos invisibles a los ejes
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        
        #Se muestra la leyenda con un punto de cada color
        ax.legend(numpoints=1)
        
        #Creamos un objeto html para exportarlo a la web
        html = mpld3.fig_to_html(fig)
        
        return html
        
        
    
    ######################
    #   MAIN PIPELINE    #
    ######################    
    def classify_documents(self):
        #Save actual time to calculate processing time
        start_time = time.time()
        
        #Cargamos y limpiamos los datasets
        titles, texts, ranks = self.get_data()
        
        #Tokenizamos y aplicamos stemming a los textos
        vocab_frame = self.tokenize_stem_texts(texts)
        
        #Vectorizamos los textos
        terms, dist, tfidf_matrix = self.vectorize_texts_tfidf(texts)
        
        #Aplicamos el clustering
        clusters, km = self.apply_kmeans(self.NUM_CLUSTERS, tfidf_matrix)       
        
        #Creamos un dataframe con los campos extraidos
        frame = self.create_dataframe(titles, ranks, texts, clusters)
        
        #Contamos los documentos de cada cluster
        num_docs_0, num_docs_1 = self.count_clusters(frame)
        
        #Obtenemos las palabras clave y los títulos de los documentos de cada cluster
        cluster0_words, cluster0_titles, cluster1_words, cluster1_titles = self.get_top_terms_titles(self.NUM_CLUSTERS, km, vocab_frame, terms, frame)
                        
        #Reducimos la dimensionalidad para la visualizacion
        xs, ys = self.reduce_dimensions(dist)
        
        # img_html = self.visualize_results(cluster0_words, cluster1_words, xs, ys, clusters, titles)
        
        #Calculate processing time
        proc_time = time.time()-start_time
        
        final_frame = {
                       # 'img':img_html, 
                       '0_words':cluster0_words, 
                       '1_words':cluster1_words, 
                       '0_titles':cluster0_titles,
                       '1_titles':cluster1_titles,
                       '0_docs':num_docs_0,
                       '1_docs':num_docs_1,
                       'time':proc_time
                       }
    
        return final_frame

    def printcwd(self):
        return os.getcwd()

#class TopToolbar(mpld3.plugins.PluginBase):
#    """Plugin de D3 para mover la bara de herramientas a la parte de arriba de la figura"""
#
#    JAVASCRIPT = """
#    mpld3.register_plugin("toptoolbar", TopToolbar);
#    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
#    TopToolbar.prototype.constructor = TopToolbar;
#    function TopToolbar(fig, props){
#        mpld3.Plugin.call(this, fig, props);
#    };
#
#    TopToolbar.prototype.draw = function(){
#      // the toolbar svg doesn't exist
#      // yet, so first draw it
#      this.fig.toolbar.draw();
#
#      // then change the y position to be
#      // at the top of the figure
#      this.fig.toolbar.toolbar.attr("x", 150);
#      this.fig.toolbar.toolbar.attr("y", 400);
#
#      // then remove the draw function,
#      // so that it is not called again
#      this.fig.toolbar.draw = function() {}
#    }
#    """
#    def __init__(self):
#        self.dict_ = {"type": "toptoolbar"}



########################
#SIMULACIÓN DE EJECUCIÓN
########################
        
#PATH = './data/'
#TITLES_NAME = 'titles.txt'
#TEXTS_NAME = 'texts.txt'
#cluster_obj = Cluster(num_clusters=2)  
#
#df_final = cluster_obj.classify_documents(PATH, TITLES_NAME, TEXTS_NAME)
        
