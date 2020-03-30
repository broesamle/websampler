import codecs
import html
import logging
import os
import re

import nltk.tokenize as tok
import scrapy

import infstruments.classify as cla

class BroadbandSpider(scrapy.Spider):
    """ Experimental generic spider to extract sentences of natural language.

    USAGE:
    $ scrapy crawl sentences_broadband -a url=https://en.wikipedia.org/wiki/Radio -o radio.json --logfile crawl.log
    """
    name = "sentences_broadband"
    TAKE = "_TAKE_"
    DROP = "_DROP_"
    OPEN = "_OPEN_"
    SPIDER_SPECIFIC_DEFAULTS = {
                'url': "https://en.wikipedia.org/wiki/Band-pass_filter",
                'trainpath': "training",
                'code': "code_aa.txt",
                'nlang': "nlang-de_aa.txt",
                'unclear': "json_aa.txt"
            }
    def __init__(self, *args, **kwargs):
        def _preproc(s):
            """ Preprocessing string content for character frequency analysis.

            Map string content so that the character frequency in the resulting
            string makes it easy to distinguish program code from natural
            language.
            """
            s = re.sub('[#@^]', '@', s) # Special chars in natural language
            s = re.sub(r'\d', '#', s)   # Digits
            s = re.sub(r'\w', 'L', s)   # Characters (digits already replaced)
            ### program language related specials
            s = re.sub(r'===|!==|\(\);', 'ccc', s)  # 3 char operators
            ### Typical elements in code: () && || ... =" !=
            s = re.sub(r'\(\)|&&|\|\||\+\+|--|[-+!=<>]=|!!|=[\'"]', 'cc', s)
            s = re.sub(r'[<>|@/\\{}\[\]()]', ']', s)  # braces
            return s
        logging.info("sentences_broadband: __init__")
        logging.info("args:%s kwargs:%s" % (args, kwargs))
        self.spsp_settings = BroadbandSpider.SPIDER_SPECIFIC_DEFAULTS
        self.spsp_settings.update(kwargs)
        BroadbandSpider.start_urls = [self.spsp_settings['url']]
        # load training examples for programming code
        fname = os.path.join(self.spsp_settings['trainpath'],
                             self.spsp_settings['code'])
        f = codecs.open(fname, "r", encoding="utf-8")
        code = [ re.sub(r"[\n\r]", "", c) for c in f.readlines() ]
        f.close()
        # load training examples for natural language
        fname = os.path.join(self.spsp_settings['trainpath'],
                             self.spsp_settings['nlang'])
        f = codecs.open(fname, "r", encoding="utf-8")
        nlang = [ re.sub(r"[\n\r]", "", c) for c in f.readlines() ]
        f.close()
        # load training examples for cases to keep open/undecided
        fname = os.path.join(self.spsp_settings['trainpath'],
                             self.spsp_settings['unclear'])
        f = codecs.open(fname, "r", encoding="utf-8")
        unclear = [ re.sub(r"[\n\r]", "", c) for c in f.readlines() ]
        f.close()
        training = code + nlang + unclear
        target = ( [BroadbandSpider.DROP]*len(code)
                 + [BroadbandSpider.TAKE]*len(nlang)
                 + [BroadbandSpider.OPEN]*len(unclear))
        self.clfier = cla.CharStatsClassifier(
                    training, target, preproc=_preproc, debuglog=True)
        super().__init__(*args, **kwargs)

    def parse(self, response):
        logging.info("parse: %s" % response.url)
        for h1 in response.css('h1::text').getall():
            yield { 'h1': h1 }
        all_texts = []
        for text in response.css('body *::text').getall():
            text = html.unescape(text.strip())
            text = re.sub("\s+", " ", text)
            if text == "": continue
            if len(text) < 10:
                logging.debug("P: _SMALL_ %s" % text)
                all_texts.append(text)
            else:
                cla = self.clfier.classify_s(text)
                if cla in [BroadbandSpider.DROP, BroadbandSpider.OPEN]:
                    all_texts.append("-X-")
                else:
                    all_texts.append(text)
        sentences = tok.sent_tokenize(" ".join(all_texts))
        for sent in sentences:
            yield {'sentence': sent}
