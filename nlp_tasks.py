# -*- coding: utf-8 -*-
#! /usr/bin/python3
import MeCab
from sklearn.feature_extraction.text import CountVectorizer

def _split_to_words(text):
	tagger = MeCab.Tagger('-O wakati')
	try:
		res = tagger.parse(text.strip())
	except:
		return[]
	return res

def get_vector_by_text_list(_items):
	count_vect = CountVectorizer(analyzer=_split_to_words)
	box = count_vect.fit_transform(_items)
	X = box.todense()
	return [X,count_vect]