# import pickle
#
# # 加载协议5的pickle文件
# with open(r"E:\医疗检测系统\cantrip-master\cantrip-master\data\haa\test.pickle", "rb") as f:
#     data = pickle.load(f)
# with open("test.pkl", "wb") as f:
#     pickle.dump(data, f, protocol=4)
from collections import Counter

nums=[1,1,1,1,1]
k=10

res=0
for i in range(0,len(nums)):
    cur=0
    a = Counter()
    for j in range(i,len(nums)):
        if a[nums[j]]+1>=2:
            cur+=a[nums[j]]
        a[nums[j]]+=1
        if cur>=k:
            res+=len(nums)-j
            break
print(res)

'''
 python run_experiment.py --data_dir "E:\医疗检测系统\cantrip-master\cantrip-master\data\haa"  --output_dir "E:\医疗检测系统\cantrip-master\cantrip-master\data\haa"
'''