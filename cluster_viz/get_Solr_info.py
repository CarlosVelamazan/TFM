# -*- coding: utf-8 -*-
"""
@author: Carlos Velamazán Mas
"""

import pysolr
import os

class SolrSearch():
    def __init__(self, key_word):
        self.word = key_word        
        self.out_dir = './data/'
    
    def get_docs(self):
        # try:    
            #Nos conectamos con solr a traves del puerto de la maquina virtual
            solr = pysolr.Solr('http://192.168.79.129:8983/solr/news_english/')
            
            #Buscamos la palabra que nos llega por el formulario
            results = solr.search(self.word, rows = 100)
            
            #Iteramos sobre los resultados y construimos los datasets
            titles = []
            texts = []
            for result in results:
                titles.append(result['title'])
                texts.append(result['text'])
            print(titles)
            return titles, texts
        
            #Creamos los ficheros con la informacion
            # f_titles = open(self.out_dir+'titles.txt', 'wb')
            # f_texts = open(self.out_dir+'texts.txt', 'wb')
            
            # #Escribimos los datasets en los ficheros
            # for title in titles:
            #     f_titles.write("%s\nNUEVOTEXTO\n" % title)
            # for text in texts:
            #     f_texts.write("%s\nNUEVOTEXTO\n" % text)
                
            #Cerramos los ficheros
            # f_titles.close()
            # f_texts.close()

            

        #     return "OK"
        # except:
        #     return "Algo ha salido mal..."


#busqueda = SolrSearch('refugees')
#busqueda.get_docs()



