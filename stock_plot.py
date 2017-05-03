import datetime
import urllib
import pandas as pd
import matplotlib.pyplot as plt

class Quote(object):

  DATE_FMT = '%Y-%m-%d'
  TIME_FMT = '%H:%M:%S'

  def __init__(self):
    self.symbol = ''
    self.date,self.time,self.open_,self.high,self.low,self.close,self.volume = ([] for _ in range(7))

  def append(self,dt,open_,high,low,close,volume):
    self.date.append(dt.date())
    self.time.append(dt.time())
    self.open_.append(float(open_))
    self.high.append(float(high))
    self.low.append(float(low))
    self.close.append(float(close))
    self.volume.append(int(volume))

  def to_csv(self):
    return ''.join(["{0},{1},{2},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7}\n".format(self.symbol,
              self.date[bar].strftime('%Y-%m-%d'),self.time[bar].strftime('%H:%M:%S'),
              self.open_[bar],self.high[bar],self.low[bar],self.close[bar],self.volume[bar])
              for bar in xrange(len(self.close))])

  def write_csv(self,filename):
    with open(filename,'w') as f:
      f.write(self.to_csv())

  def read_csv(self,filename):
    self.symbol = ''
    self.date,self.time,self.open_,self.high,self.low,self.close,self.volume = ([] for _ in range(7))
    for line in open(filename,'r'):
      symbol,ds,ts,open_,high,low,close,volume = line.rstrip().split(',')
      self.symbol = symbol
      dt = datetime.datetime.strptime(ds+' '+ts,self.DATE_FMT+' '+self.TIME_FMT)
      self.append(dt,open_,high,low,close,volume)
    return True

  def __repr__(self):
    return self.to_csv()


# In[8]:

class YahooQuote(Quote):
  ''' Daily quotes from Yahoo. Date format='yyyy-mm-dd' '''
  def __init__(self,symbol,start_date,end_date=datetime.date.today().isoformat()):
    super(YahooQuote,self).__init__()
    self.symbol = symbol.upper()
    start_year,start_month,start_day = start_date.split('-')
    start_month = str(int(start_month)-1)
    end_year,end_month,end_day = end_date.split('-')
    end_month = str(int(end_month)-1)
    url_string = "http://ichart.finance.yahoo.com/table.csv?s={0}".format(symbol)
    url_string += "&a={0}&b={1}&c={2}".format(start_month,start_day,start_year)
    url_string += "&d={0}&e={1}&f={2}".format(end_month,end_day,end_year)
    csv = urllib.urlopen(url_string).readlines()
    csv.reverse()
    for bar in xrange(0,len(csv)-1):
      ds,open_,high,low,close,volume,adjc = csv[bar].rstrip().split(',')
      open_,high,low,close,adjc = [float(x) for x in [open_,high,low,close,adjc]]
      if close != adjc:
        factor = adjc/close
        open_,high,low,close = [x*factor for x in [open_,high,low,close]]
      dt = datetime.datetime.strptime(ds,'%Y-%m-%d')
      self.append(dt,open_,high,low,close,volume)


if __name__ == '__main__':
    q = YahooQuote('UAL','2017-04-01')        # download year to date United Airline data
    q.write_csv('UA.csv')                     # save it to disk
    #q = Quote()                              # create a generic quote object
    #q.read_csv('UA.csv')                     # populate it with our previously saved data

    df = pd.read_csv('UA.csv', header=None)
    df.columns = ["Company", "Date", "Time", "Open Price", "Highest Price", "Lowest Price", "Close Price", "Volume"]
    df = df.drop('Time', 1)
    df['Price Change'] = df['Close Price'] - df['Open Price']
    df['Price Movement'] = ""
    for i in range(0,len(df['Price Change'])):
        if df['Price Change'][i] > 0:
            df.set_value(i, 'Price Movement', "Up")
        elif df['Price Change'][i] < 0:
            df.set_value(i, 'Price Movement', "Down")
        else:
            df.set_value(i, 'Price Movement', "Not changed")


    Price_Change = df['Price Change'].tolist()
    Date = df['Date'].tolist()
    for i in range(0, len(Date)):
        Date[i] = Date[i].replace('-', '.')
    print df


    #Plot
    x = [i for i in range(0, len(Date))]
    plt.xticks(x, Date, rotation = 90)
    y2 = [0]*len(Date)
    #plt.yticks(y, Price_Change)
    plt.plot(x, Price_Change, 'ro', x, y2)
    plt.show()
