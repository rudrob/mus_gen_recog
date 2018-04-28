import urllib.request
from bs4 import BeautifulSoup
import time
from corpus_artists import artists_genres

BASE_SONG_URL = "http://www.tekstowo.pl/piosenka,"
BASE_ARTIST_URL = "http://www.tekstowo.pl/piosenki_artysty,"
SONGS_PER_ARTIST = 10
DELAY_BETWEEN_EACH_SONG_DL = 1


def create_files_for_artist(artist, logfile):
    postfix = ",popularne,malejaco,strona,1.html"
    with urllib.request.urlopen(BASE_ARTIST_URL + artist + postfix) as response:
        html_doc = response.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        if '/piosenka,' + artist in a['href']:
            links.append(a['href'])
            if len(links) == SONGS_PER_ARTIST:
                break

    time.sleep(DELAY_BETWEEN_EACH_SONG_DL)

    for link in links:
        urltosong = BASE_SONG_URL + link[10:]
        with urllib.request.urlopen(urltosong) as response:
            html_doc = response.read()
            soup = BeautifulSoup(html_doc, 'html.parser')
            div = soup.findAll("div", {"class": "song-text"})
            songname = link.replace("/", "_").replace(",", "-")[10:]
            songfile = open("corpus/" + artists_genres[artist] + "/" + songname + ".txt", "w+")
            logfile.write("Writing song: " + songname + "\n")
            song_text = div[0].get_text()
            song_text = song_text.replace("Tekst piosenki:", "")
            song_text = song_text.replace("Poznaj historię zmian tego tekstu", "")
            songfile.write(song_text)
        time.sleep(1)

def create_corpus():
    logfile = open("corpus/logfile.txt", "w+")
    for artist in artists_genres.keys():
        create_files_for_artist(artist, logfile)
    logfile.close()




