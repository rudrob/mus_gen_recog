''' FOR AZ LYRICS BUT THEY BANNED MY IP LOL
BASEURL = "http://www.azlyrics.com/"

def create_files_for_artist(artist_postfix):
    logfile = open("corpus/logfile.txt", "w+")
    with urllib.request.urlopen(BASEURL + artist_postfix) as response:
        html_doc = response.read()
    soup = BeautifulSoup(html_doc, 'html.parser')

    links = []
    for all_a in soup.find_all('a', href=True):
        if '../lyrics' in all_a['href']:
            logfile.write("Found the URL:" + all_a['href'] + "\n")
            links.append(all_a['href'])

    for link in links:
        with urllib.request.urlopen(BASEURL + link[3:]) as response:
            html_doc = response.read()
            soup = BeautifulSoup(html_doc, 'html.parser')
            divs = soup.find_all('div')
            songname = link[10:].replace("/", "_")
            songfile = open("corpus/metal/" + songname + ".txt", "w+")
            logfile.write("Writing song: " + songname + "\n")
            songfile.write(divs[21].get_text())
            if(len(links) == 10):
                break
            # print(soup.find_all('div')[11].get_text())

    logfile.close()

create_files_for_artist("s/sabaton.html")
'''