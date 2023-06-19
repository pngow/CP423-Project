import os, argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import requests
import justext
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# https://web.ics.purdue.edu/~gchopra/class/public/pages/webdesign/05_simple.html
# http://books.toscrape.com/

class WebCrawler:

    PAGES_PER_TOPIC = 100

    def get_init_urls(self):
        # NOTE: LIFO / FIFO?
        self.urls = OrderedDict()
        self.topics = set()

        # read all source URLs from file
        with open('sources.txt', 'r') as f:
            line = f.readline()

            while line:
                # 0 = topic, 1 = url
                tokens = line.split(',')
                self.urls[tokens[1].strip()] = tokens[0].strip()

                # add to unique list of topics
                self.topics.add(tokens[0].strip())

                line = f.readline()

        # convert set to list to iteratte with index
        self.topics = list(self.topics)
        # get list of initial urls & their topics ... topics as keys
        self.init_urls = dict()
        for k, v in self.urls.items():
            self.init_urls[v] = self.init_urls.get(v, []) + [k]

        # create folders to store urls/content from list of topics
        self.create_folders(self.topics)

        return
    
    def get_topics(self):

        return self.topics
    
    def create_folders(self, topics):
        # get path to current folder
        folder_path = Path(__file__).parent.absolute()

        # create fodler for each unique topic from source file
        for topic in topics:
            # create path for folder
            new_path = os.path.join(folder_path, 'data', topic)

            # check if folder already exists
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            else:
                # remove and read folder
                shutil.rmtree(new_path)
                os.makedirs(new_path)
        
        return

    def download_content(self, resp):
        '''
            Downloads content from the web page of the URL given.

            Params:
                resp = response from HTTP request
            
            Return:
                content = content of the web page
                download_datetime = date & time the web page was downloaded
        '''
        stop_words = set(stopwords.words('english'))

        # download content from the web page
        content = ""

        # get download date and time
        download_datetime = datetime.now()

        paragraphs = justext.justext(resp.content, justext.get_stoplist("English"))
        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
                # this asseration makes sure we catch string and unicode only
                assert isinstance(paragraph.text, str)
                # https://portingguide.readthedocs.io/en/latest/strings.html
                # convert byte string to readable text
                if type(paragraph.text) == bytes:
                    paragraph_text = paragraph.text.decode('utf8', 'ignore')
                else:
                    paragraph_text = paragraph.text

                # tokenize text
                paragraph_tokens = regexp_tokenize(paragraph_text, r'\s+', gaps=True)

                # remove stop words
                filtered_paragraph = [w for w in paragraph_tokens if not w.lower() in stop_words]
                # join string back together
                filtered_paragraph = ' '.join(filtered_paragraph)

                # add to final content string for the file
                content += str(filtered_paragraph) + "\n"

        # print(content)

        return content, download_datetime

    def write_content(self, topic, url_hash, content):
        '''
            Writes content from the web page of the URL given to a txt file.
            Name of the file is the hash value of the URL. 

            Params:
                resp = response from HTTP request

            Return:
                N/A
        '''
        # write downloaded content to a file with the url's hash as the name 
        # encode to utf-8
        outFile = open(f'data/{topic}/{url_hash}.txt', mode='w', encoding="utf-8")
        outFile.write(content)
        outFile.close()

        return

    def get_urls(self, topic, url, html):
        '''
            Scrape URLs to traverse to on next on the web page.

            Params:
                url = the initial url given
                html = the html of the web page

            Return:
                N/A
        '''
        soup = BeautifulSoup(html, 'html.parser')
        # find all instances of the hyperlink tag to obtain all links ('a' tags with hred attribute) on the web page
        for link in soup.find_all('a', href=True):
            # get the url from the href attribute in the tag
            path = link.get('href')

            if (path == "" or path == None or urlparse(path).hostname == None):
                continue

            # account for relative urls below (start with /...)
            if path and path.startswith('/'):
                # concat the relative path to the intial url
                path = urljoin(url, path)
            # elif path and (not path.startswith('http') or not path.startswith('www')):
            #     path = urljoin(url, path)

            host_name = urlparse(path).scheme + '://' + urlparse(path).hostname + '/'
            
            # link exactly includes initial URL, crawl it (step 1). Otherwise, ignore it. It helps your search being focused on your selected topics only
            # add /ca for sites given that were specifically for Canada
            if host_name in self.init_urls[topic] or host_name + '/ca' in self.init_urls[topic]:
                # 
                yield path


    def create_log_file(self, topic, url_hash, url, dl_datetime):
        '''
            Create and write/append to the log file. Contains all URLs travelled to, the hash value, download date/time, and HTTP response code.

            Params:
                url_hash = hash value of the url
                url = url given
                dl_datetime = date/time of when the web page was downloaded
                http_status = http status of the web page

            Return:
                N/A
        '''
        # create output string for log file
        out_str = f"<{topic}, {url}, {url_hash}, {dl_datetime}>\n"

        outFile = open('crawl.log', mode='a', encoding="utf-8")
        outFile.write(out_str)
        outFile.close()

        return

    def create_mapping(self, topic, doc_num, url_hash):
        # create output string for map file
        out_str = f"{topic}/{url_hash}, H{doc_num}\n"

        outFile = open('mapping.txt', mode='a', encoding="utf-8")
        outFile.write(out_str)
        outFile.close()

        return

    # NOTE: HOW MANY URLS PER TOPIC? 100 OR SPLIT THE 100?
    def check_num_files(self):
        # want folders to have at least 100 urls each
        flag = False
        i = 0

        # check number of files in each topic folder
        while i < len(self.topics):
            count = 0

            # iterate through list of files in topic directory
            for file in os.listdir('data/' + self.topics[i]):
                # check if file path exists to the file listed in the directory
                if os.path.isfile(os.path.join('data', self.topics[i], file)):
                    count += 1

            # check if final count was more than 100
            if count < self.PAGES_PER_TOPIC:
                flag = True
            else:
                # remove urls related to that topic if the count of files > 100 (don't need to crawl anymore)
                if self.topics[i] in self.urls.values():
                    self.urls = OrderedDict({key:val for key,val in self.urls.items() if val != self.topics[i]})

            # print(f"{self.topics[i]} {count}")
            i += 1

        return flag

    def run(self):
        print("\nStarting crawl.")

        self.get_init_urls()
        # iterator for loop
        num_urls = 1

        while (self.urls and self.check_num_files()):             
            
            # NOTE: LIFO / FIFO ?
            url_info = self.urls.popitem(last=False)
            url = url_info[0]
            topic = url_info[1]

            try:
                # http request to the given url page
                resp = requests.get(url)
                # get hash value for the url
                url_hash = hashlib.sha256(url.encode()).hexdigest()

                # check HTTP status?
                if resp.status_code == 200:              

                    # PART 1b
                    # no hash file (not read yet) 
                    if not os.path.exists(f"data/{topic}/{url_hash}.txt"):
                        # PART 4
                        # crawl link
                        for link in self.get_urls(topic, url, resp.text):
                            # set topic of link being crawled
                            if link not in self.urls:
                                self.urls[link] = topic

                        # PART 1a / 1c
                        # download content from the url 
                        content, download_datetime = self.download_content(resp)

                        # no text from download, skip the page
                        # NOTE: how much text do we want to keep?
                        if content == "" or len(content) < 2500:
                            continue

                        # PART 2
                        self.write_content(topic, url_hash, content)

                        # if first url being processed
                        if num_urls == 1:
                            # delete file if exists ... refresh the log
                            if os.path.exists("crawl.log"):
                                os.remove("crawl.log")
                            
                            # create mapping file for part 2
                            if os.path.exists("mapping.txt"):
                                os.remove("mapping.txt")
                        # create mapping for doc ids and hash for inverted index
                        self.create_mapping(topic, num_urls, url_hash)
                        # create log for the web page crawled
                        self.create_log_file(topic, url_hash, url, download_datetime)

                        # print(f"{url} {num_urls}")
                        num_urls += 1
                else:
                    print(f"HTTP code: {resp.status_code} for URL: {url}. Could not crawl.")
            except Exception as e:
                print(e)

        print("Crawl complete.\n")

if __name__ == '__main__':

    WebCrawler().run()