import os
import json
import pandas as pd
'''
the following function creates a map {dd/mm/hh : {'TerminateInstances': 0,'ConsoleLogin':0 ,'CreateTags':0,'RebootInstances' : 0,'Errors':0}}}

'''
def year(map):
	mcount =1
	t = 0
	for m1 in range(2):
		for m2 in range(10):
			dcount = 0
			if mcount ==13:
				break
			if m1==0 and m2 ==0:
				continue
			for d1 in range(4):
				for d2 in range(10):
					hcount = 0
					if d1 == 0 and d2 == 0:
						continue
					if (mcount==2 and dcount ==28):
						break
					elif ((mcount ==1 or mcount==3 or mcount==5 or mcount==7 or mcount==8 or mcount==10 or mcount==12) and dcount ==31):
						break
					elif ((mcount==4 or mcount==6 or mcount==9 or mcount==11) and dcount ==30):
						break
					# if dcount ==0:
					# 	continue
					for h1 in range(3):
						for h2 in range(10):
							if hcount ==24:
								break
							map['%d%d/%d%d/%d%d' %(m1,m2,d1,d2,h1,h2)] = {'TerminateInstances': 0,'ConsoleLogin':0 ,'CreateTags':0,'RebootInstances' : 0,'Errors':0}
							hcount +=1
							t +=1
					dcount +=1
			mcount +=1m

def times_so_far(ls):
	countList = []
	keyList = []
	for i in ls:
		if i not in keyList:
			keyList.append(i)
	for i in keyList:
		countList.append(ls.count(i))
	return keyList, countList

if __name__ == "__main__":
	mymap = {}
	year(mymap)
	flag = 0
	eventList = []
	errorList = []
	newdict ={}
	dict = {}
	hourSet = set()
	RecordDict = {}

	eventList = ['TerminateInstances','ConsoleLogin','CreateTags','RebootInstances']

	# for i in errorList:
	# 	print(i)

'''
following code reads the s3 logs from log files and find events and errors and append it to its respective date in the map
'''
	for root, dirs, files in os.walk('F:\\MS\\2015 cloudtrail\\2015'):
		count = 0
		mdd = list()
		m = ''
		d = ''
		h = ''

		for f in files:
			if f.endswith('.'+'json'):
				file = open(os.path.join(root,f))
				records = file.readlines()
				file.close
				x = 0
				for r in records:
					myjson = json.loads(r)
					for p in range(len(myjson['Records'])-1):
						m = myjson['Records'][p]['eventTime'][5:7]
						d = myjson['Records'][p]['eventTime'][8:10]
						h = myjson['Records'][p]['eventTime'][11:13]
						if 'eventName' not in myjson['Records'][p]:
							continue
						else:
							for i in eventList:
								if i == myjson['Records'][p]['eventName']:
									mymap[m+'/'+d+'/'+h][i] += 1
		for f in files:
			if f.endswith('.'+'json'):
				file = open(os.path.join(root,f))
				records = file.readlines()
				file.close
				x = 0
				for r in records:
					myjson = json.loads(r)
					for p in range(len(myjson['Records'])-1):
						m = myjson['Records'][p]['eventTime'][5:7]
						d = myjson['Records'][p]['eventTime'][8:10]
						h = myjson['Records'][p]['eventTime'][11:13]
						if 'errorCode' not in myjson['Records'][p]:
							continue
						else:
							mymap[m+'/'+d+'/'+h]['Errors'] += 1


	# print(mymap)
	#following code creates a csv file of the map, which will serve as the input to the lstm model

	df = pd.DataFrame(mymap).T
	df.to_csv('F:\\MS\\Cloudtrail\\lstminput.csv',sep = ',' )
	# print(df)


