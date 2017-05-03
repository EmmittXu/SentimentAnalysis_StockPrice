import socket
from threading import Thread
import time as tm
import getTweet
import csv

try:
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error, msg:
	print ">>>Failed to creat Socket. Error Code:" + str(msg[0]) + " Message:" + msg[1] + '.'
	sys.exit()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('localhost',9999))
s.listen(1)
con, addr = s.accept()

negative=0.0
neutral=0.0
positive=0.0

api=getTweet.TwitterClient()
def sending():
	#Stream tweets in real-time and send data to the port
	while(True):
		i=0
		while i<=100:
			tm.sleep(0.02)
			tweets = api.get_tweets(query = "Apple", count = 1)
			for tweet in tweets:
				message = tweet
				con.send(message + "\n")
			i+=1

	#Thi is just an alternative way to do the streaming demo if access denied by Twitter API 
	#It reads a local file and send data to the port
	# while(i==0):
	# 	with open("/home/emmittxu/Desktop/Stock-Sentiment-alalysis/testdata.manual.2009.06.14.csv") as f:
	# 		lines=csv.reader(f)
	# 		for line in lines:
	# 			message=line[5]
	# 			tm.sleep(0.1)
	# 			con.send(message+"\n")


def main():
	try:
		thread_2 = Thread(target = sending, args = ())
		thread_2.setDaemon(True)
		thread_2.start()
	except:
		print "Unable to start thread"

if __name__ == '__main__':
	sending()
