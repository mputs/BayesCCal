import requests

URLS = [("data_banknote_authentication.txt", "https://www.dropbox.com/s/dl/wxszlv7slv4854h/data_banknote_authentication.txt?dl=1"), ("SUSY.csv", "https://www.dropbox.com/s/p5cam3bi1zj6frc/SUSY.csv?dl=1")]

for fnam, url in URLS:
	print(fnam, url)
	r = requests.get(url, allow_redirects=True)
	open(fnam, "wb").write(r.content)

