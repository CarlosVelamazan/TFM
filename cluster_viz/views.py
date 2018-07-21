from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .models import Post
from .forms import SearchForm
from .execute_cluster import Cluster
from .get_Solr_info import SolrSearch

def main(request):
    # posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'cluster_viz/search.html')

def cluster_detail(request):
    word = request.GET.get('q', 'jaguar')
    if word != '':
        word = request.GET['q']
    else:
        word = 'Jaguar'

    solr = SolrSearch(word)
    titles, texts = solr.get_docs()
    word = word.capitalize()
    # PATH = './cluster_viz/data/'
    # TITLES_NAME = 'titles.txt'
    # TEXTS_NAME = 'texts.txt'
    cluster_obj = Cluster(titles, texts)  
    df_final = cluster_obj.classify_documents()
    words0 = ', '.join(df_final['0_words'])
    words1 = ', '.join(df_final['1_words'])
    titles0 = df_final['0_titles']
    titles1 = df_final['1_titles']
    docs0 = df_final['0_docs']
    docs1 = df_final['1_docs']
    totaldocs = docs0+docs1
    time = df_final['time']

    return render(request, 'cluster_viz/cluster_detail.html', 
	    				  {'word': word, 'words0':words0, 'words1':words1,
	    				   'titles0': titles0, 'titles1':titles1,
	    				   'docs0': docs0, 'docs1':docs1, 'time':time, 'totaldocs':totaldocs})

def search(request):
    form = SearchForm(request.GET)

    # return HttpResponseRedirect(message)
    # return render(request, 'cluster_viz/search.html', {'word': word})
    # url = reverse('home', kwargs={'word': word})
    # return HttpResponseRedirect(url)
    return render(request, 'cluster_viz/search_edit.html', {'form': form})
