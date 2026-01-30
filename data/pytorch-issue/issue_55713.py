try:
    item = self.load_pickle("package", "may_not_exist")
except Exception as e:
    print(e)
item = self.load_pickle("package", "definitely_exists")