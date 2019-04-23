import pandas as pd 
from pymongo import MongoClient

class connectDB:
	def __init__(self):
		client = MongoClient("db_ip",port = "db_port")
		self.db = client.admin
		self.data = {}
	def read_data(self,file_path):
		try:
			d = {}
			data = pd.read_excel(file_path,sheet_name = 0)
			header = list(data.columns)
			for i in range(len(header)):
				a = list(data[header[i]])
				d[header[i]] = list(data[header[i]])
			d = dict(d)
			self.data.update(d)
		except Exception as e:
			print("Exception as: {}".format(e))

	def insert_data_to_db(self):
		try:
			result = self.db.reviews.insert_one(self.data)
			print('Data is successfully inserted with id :{}'.format(result.inserted_id))
		except Exception as e:
			print("Exception occured as :{}".format(e))

	def read_from_db(self,index):
		try:
			data = self.db.reviews.find_one(index)
			print(data)
		except Exception as e:
			print("Exception occured as :{}".format(e))


if __name__ =="__main__":
	file_loc = r"file_path"
	db = connectDB()
	db.read_data(file_path = file_loc)
	db.insert_data_to_db()
	db.read_from_db({'Hobby':'Cricket'}) #dict like format





