import matplotlib.pyplot as plt

for point in open("./tisv_model/cos_similarity.txt"):
	position = float(point.split()[0])
	flag = point.split()[1]
	if flag == 'match':
		plt.plot(position,position,'ro')
	elif flag == 'unmatch':
		plt.plot(position,position,'bo')
plt.show()