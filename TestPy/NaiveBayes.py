'''
Original Data Set:
Example No. Color Type Origin Stolen?
1 Red Sports Domestic Yes
2 Red Sports Domestic No
3 Red Sports Domestic Yes
4 Yellow Sports Domestic No
5 Yellow Sports Imported Yes
6 Yellow SUV Imported No
7 Yellow SUV Imported Yes
8 Yellow SUV Domestic No
9 Red SUV Imported No
10 Red Sports Imported Yes
Example query:
Red Domestic SUV : 010
Mapping:
Red = 0, Yellow = 1
Sports = 0, SUV = 1
Domestic = 0, Imported = 1
Yes = 0, No = 1
'''

att_0 = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
att_1 = [0, 0, 0, 0, 0, 1, 1 ,1 ,1, 0]
att_2 = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
tgt = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
p = 0.5
m = 3
data_set = zip(att_0, att_1, att_2, tgt)

def y_n(tgt):
    y = 0
    n = 0
    for i in tgt:
        if i == 1:
            y += 1
        else:
            n += 1
    return y, n

def attry_n(attr, y, n):
    attlist_0 =  [0, 0]
    attlist_1 = [0, 0]
    for index, att in enumerate(attr):
        if tgt[index] == 0:
            attlist_0[att] += 1
        else:
            attlist_1[att] += 1

    for i in range(2):
        attlist_0[i] = (attlist_0[i] + (m*p)) / (y + m)
        attlist_1[i] = (attlist_1[i] + (m*p)) / (n + m)

    return attlist_0, attlist_1

y, n = y_n(tgt)

#Colour Probability
att_0_0, att_0_1 = attry_n(att_0, y, n)

#Type Probability
att_1_0, att_1_1 = attry_n(att_1, y, n)
#Origin Probability
att_2_0, att_2_1 = attry_n(att_2, y, n)

query = list(input())
pred_y = y / (y + n)
pred_n = n / (y + n)

for index, att in enumerate(query):
    if index == 0:
        if att == 0:
            pred_y *= att_0_0[0]
            pred_n *= att_0_0[1]
        else:
            pred_y *= att_0_1[0]
            pred_n *= att_0_1[1]
    elif index == 1:
        if att == 0:
            pred_y *= att_1_0[0]
            pred_n *= att_1_0[1]
        else:
            pred_y *= att_1_1[0]
            pred_n *= att_1_1[1]
    else:
        if att == 0:
            pred_y *= att_2_0[0]
            pred_n *= att_2_0[1]
        else:
            pred_y *= att_2_1[0]
            pred_n *= att_2_1[1]

print("Predict Yes Probability: ", pred_y,
      "\nPredict No Probability: ", pred_n)

if pred_y > pred_n:
    print("Predicted Yes")
elif pred_n > pred_y:
    print("Predicted No")
else:
    print("Equally possible")