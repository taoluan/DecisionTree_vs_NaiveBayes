import pandas as pd 

dt = pd.read_csv('play_tennis.csv')
print (dt)
###Tim gia trị OutL của ptu Play = Yes
dt.Outlook[dt.Play=="Yes"]
###thong ke tan so xuat hien
dtOy = dt.Outlook[dt.Play=="Yes"]
###so lan xuat hien cua moi gtri t.t OutL
P1_1 = dtOy.value_counts()/dtOy.count()
#print(P1_1)
###Tim gia trị OutL của ptu Play = No
dtOn = dt.Outlook[dt.Play=="No"]
P1_2 = dtOn.value_counts()/dtOn.count()
P1_2 = P1_2/dtOn.count()
#print(P1_2)
###Tim gia trị OutL của ptu Temp = Yes
dtTy = dt.Temp[dt.Play=='Yes']
P2_1 = dtTy.value_counts()/dtTy.count()
###Tim gia trị OutL của ptu Temp = No
dtTn = dt.Temp[dt.Play=='No']
P2_2 = dtTn.value_counts()/dtTn.count()
#print(P2_1)
#print(P2_2)
###Tim gia trị OutL của ptu Humidity = Yes
dtHy = dt.Humidity[dt.Play=='Yes']
P3_1 = dtHy.value_counts()/dtHy.count()
###Tim gia trị OutL của ptu Humidity = No
dtHn = dt.Humidity[dt.Play=='No']
P3_2 = dtHn.value_counts()/dtHn.count()
#print(P3_1)
#print(P3_2)
###Tim gia trị OutL của ptu Humidity = Yes
dtWy = dt.Windy[dt.Play=='Yes']
P4_1 = dtWy.value_counts()/dtWy.count()
###Tim gia trị OutL của ptu Humidity = No
dtWn = dt.Windy[dt.Play=='No']
P4_2 = dtWn.value_counts()/dtWn.count()
#print(P4_1)
#print(P4_2)
Play = dt.Play.value_counts()/dt.Play.count()
#print(Play)
###Dự Báo nhãn cho Dữ Liệu
P_yes = P1_1[1]*P2_1[1]*P3_1[1]*P4_1[0]*Play[0]
P_no = P1_2[1]*P2_2[1]*P3_2[1]*P4_2[0]*Play[1]
print(P_yes)
print(P_no)
PY = P_yes/(P_no+P_yes)
PN = P_no/(P_no+P_yes)
print(PY)
print(PN)
