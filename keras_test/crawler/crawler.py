#!/usr/bin/python
# coding=UTF-8

import os
import re
import sys
import time
import threading
import requests
from urlparse import urljoin
from urlparse import urlparse
from bs4 import BeautifulSoup

def strip_url(url):
  if url == None or len(url) == 0:
    return ''

  url = re.sub('(^http\:\/\/)|(^https\:\/\/)', '', url)
  url = re.sub('^www\.', '', url)
  if url == None or len(url) == 0:
    return ''

  if url[-1] == '/':
    url = url[:-1]
  return url.strip()

def get_main_domain(url): 
  top_level_domains = [
    'com', 'edu', 'gov', 'org', 'net', 'int', 'mil', 'ac', 'ad', 'ae', 
    'af', 'ag', 'ai', 'al', 'am', 'an', 'ao', 'aq', 'ar', 'as', 'at', 
    'au', 'aw', 'ax', 'az', 'ba', 'bb', 'bd', 'be', 'bf', 'bg', 'bh', 
    'bi', 'bj', 'bl', 'bm', 'bn', 'bo', 'bq', 'br', 'bs', 'bt', 'bv', 
    'bw', 'by', 'bz', 'ca', 'cc', 'cd', 'cf', 'cg', 'ch', 'ci', 'ck', 
    'cl', 'cm', 'cn', 'co', 'cr', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 
    'de', 'dj', 'dk', 'dm', 'do', 'dz', 'ec', 'ee', 'eg', 'eh', 'er', 
    'es', 'et', 'eu', 'fi', 'fj', 'fk', 'fm', 'fo', 'fr', 'ga', 'gb', 
    'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gl', 'gm', 'gn', 'gp', 'gq', 
    'gr', 'gs', 'gt', 'gu', 'gw', 'gy', 'hk', 'hm', 'hn', 'hr', 'ht', 
    'hu', 'id', 'ie', 'il', 'im', 'in', 'io', 'iq', 'ir', 'is', 'it', 
    'je', 'jm', 'jo', 'jp', 'ke', 'kg', 'kh', 'ki', 'km', 'kn', 'kp', 
    'kr', 'kw', 'ky', 'kz', 'la', 'lb', 'lc', 'li', 'lk', 'lr', 'ls', 
    'lt', 'lu', 'lv', 'ly', 'ma', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 
    'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 
    'mv', 'mw', 'mx', 'my', 'mz', 'na', 'nc', 'ne', 'nf', 'ng', 'ni', 
    'nl', 'no', 'np', 'nr', 'nu', 'nz', 'om', 'pa', 'pe', 'pf', 'pg', 
    'ph', 'pk', 'pl', 'pm', 'pn', 'pr', 'ps', 'pt', 'pw', 'py', 'qa', 
    're', 'ro', 'rs', 'ru', 'rw', 'sa', 'sb', 'sc', 'sd', 'se', 'sg', 
    'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'ss', 'st', 
    'su', 'sv', 'sx', 'sy', 'sz', 'tc', 'td', 'tf', 'tg', 'th', 'tj', 
    'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tr', 'tt', 'tv', 'tw', 'tz', 
    'ua', 'ug', 'uk', 'um', 'us', 'uy', 'uz', 'va', 'vc', 've', 'vg', 
    'vi', 'vn', 'vu', 'wf', 'ws', 'ye', 'yt', 'za', 'zm', 'zw'
  ]

  url = re.sub('\/.+$', '', url, 1)
  main_domain = re.sub('^[^\.]+\.', '', url, 1)
  if re.compile('^' + '$|^'.join(top_level_domains) + '$').match(main_domain) != None:
    return strip_url(url)
  if re.compile('^' + '\.|^'.join(top_level_domains) + '\.').match(main_domain) != None:
    return strip_url(url)
  return strip_url(main_domain)

class UrlManager():
  def __init__(self):
    self.urls = []
    self.visited_urls = {}
    self.url_counter = 1

  def push_url(self, url): 
    if not (url in self.visited_urls):
      self.urls = [(self.url_counter, url)] + self.urls
      self.visited_urls[url] = True
      self.url_counter += 1

  def pop_url(self): 
    if (len(self.urls) <= 0):
      return ""

    tmp = self.urls[-1]
    del self.urls[-1]
    return tmp

  def has_url(self): 
    return len(self.urls) > 0

  def load_from_file(self, filename):
    with open(filename) as f:
      for url in f:
        self.push_url(url.strip())

class DownloaderThread(threading.Thread):
  lock = threading.Lock()

  def __init__(self, url_manager, output_dir, thread_id):
    threading.Thread.__init__(self)
    self.url_manager = url_manager
    self.thread_id = thread_id
    self.output_dir = output_dir

  def is_subdomain(self, base_url, href): 
    url_1 = strip_url(urlparse(base_url).hostname)
    url_2 = strip_url(urlparse(href).hostname)
    return get_main_domain(url_1) == get_main_domain(url_2)

  def extract_links(self, base_url, text): 
    soup = None
    try:
      soup = BeautifulSoup(text, 'html.parser')
    except HTMLParseError:
      print 'Parse error:', base_url
      return

    for a in soup.find_all('a'):
      if not a.has_attr('href'): continue
      if len(a['href']) > 0 and a['href'][0] == "#": continue

      href = urljoin(base_url, a['href'])

      if not self.is_subdomain(base_url, href): 
        continue

      url = urlparse(href).geturl()
      self.url_manager.push_url(url)

  def run(self): 
    DownloaderThread.lock.acquire()
    print "Starting thread", self.thread_id
    DownloaderThread.lock.release()

    while (self.url_manager.has_url()):
      DownloaderThread.lock.acquire()
      base_url = self.url_manager.pop_url()
      DownloaderThread.lock.release()

      url_id = base_url[0]
      base_url = base_url[1].strip()
      proxy = base_url.replace("http://", "").replace("https://", "")
      proxies = { 'http': proxy }

      r = None
      try:
        r = requests.get(base_url, proxies = { 'http': proxy }, timeout = 5)
      except:
        DownloaderThread.lock.acquire()
        print "Failed downloading", base_url, "with id", url_id
        DownloaderThread.lock.release()
        continue;

      dir = os.path.dirname(__file__)
      filename = os.path.join(os.path.dirname(__file__), self.output_dir, str(url_id) + ".html")
      text = r.text.encode('utf8')
      with open(filename, "w") as f:
        f.write(text)

      DownloaderThread.lock.acquire()
      self.extract_links(base_url, text)
      print "Written file", base_url, "with id", url_id
      DownloaderThread.lock.release()

    sys.stdout.flush()

if __name__ == "__main__":
  start_time = time.time()

  url_manager = UrlManager()
  url_manager.push_url("https://www.soic.indiana.edu/all-people/index.html")
  url_manager.push_url("http://eng.auburn.edu/ece/faculty/")
  url_manager.push_url("https://cs.illinois.edu/people/faculty/department-faculty")
  url_manager.push_url("https://www.cics.umass.edu/people/all-faculty-staff")
  url_manager.push_url("http://ee.princeton.edu/people/faculty")
  url_manager.push_url("http://www.swosu.edu/common/people-search/public/business-computer-science.php")
  url_manager.push_url("http://www.uncw.edu/csc/about/facultystaff.html")
  url_manager.push_url("http://salisbury.edu/mathcosc/dept-directory-old.html")

  threads = []
  for i in range(0, 8):
    t = DownloaderThread(url_manager, sys.argv[1], i)
    threads.append(t)
    t.start()
    time.sleep(1)

  for t in threads:
    t.join()

  print 'The process took', time.time() - start_time, 'seconds.'
