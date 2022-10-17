import json
import jiagu
import pandas as pd

results = []
rows = []                                               # row存放key对应的内容
def read_file(dir, left, right):
    file = open(dir, encoding='utf-8')
    count = 0
    for line in file:
        if  count<left:
          count = count + 1
          continue
        if  left <= count < right:
            # key = set_default_p(line)
            # words = key.split()
            # key = " ".join(sorted(set(words), key=words.index))  # 去掉重复的单词
            key = jiagu.keywords(line, 5)
            keyword = ""
            for i in range(len(key)):
              keyword = keyword + key[i] + " "
            row = [keyword, line]
            rows.append(row)
        else:
            break
        count = count + 1
    # print(rows)
    for row in rows:
      results.append(row)
    # results = dict(zip(keys, row)) for row in rows
    # print(results)

def extract():
    file_path = "./text3.txt"
    left = 0
    right = 100000
    read_file(file_path,left,right)

extract()

data = pd.DataFrame(results, columns=["keywords", "text"])
print(data)



from keytotext import trainer
# train_df = DataFrame()
# test_df = DataFrame()
# print(test_df)

model = trainer()

model.from_pretrained(model_name="Langboat/mengzi-t5-base")
# model.from_pretrained(model_name="IDEA-CCNL/Randeng-T5-77M")
model.train(train_df=data[:10000], test_df=data[10000:13000], batch_size=4, max_epochs=3,use_gpu=False)
model.save_model()
