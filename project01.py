import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pylab as pylab

font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\malgunbd.ttf").get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

### 폰트 관련 설정 
font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\malgunbd.ttf").get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

############################################ 데이터 확인 #################################################

# 2020년 일별 평균 미세먼지 데이터 불러오기
df_mise_2020 = pd.read_csv('C:/Users/user/Desktop/일별평균대기오염도_2020.csv',encoding='cp949')
df_mise_2020

df_mise_2020.shape # (18174, 8)

df_mise_2020.columns

"""
Index(['측정일시', '측정소명', '이산화질소농도(ppm)', '오존농도(ppm)', '이산화탄소농도(ppm)',
       '아황산가스(ppm)', '미세먼지(㎍/㎥)', '초미세먼지(㎍/㎥)'],
      dtype='object')
"""

df_mise_2020.size # 145392

df_mise_2020.dtypes

"""
측정일시              int64
측정소명             object
이산화질소농도(ppm)    float64
오존농도(ppm)       float64
이산화탄소농도(ppm)    float64
아황산가스(ppm)      float64
미세먼지(㎍/㎥)       float64
초미세먼지(㎍/㎥)      float64
dtype: object
"""

df_mise_2020_copy = df_mise_2020

# 선택하여 열 이름 변경하기
df_mise_2020_copy = df_mise_2020_copy.rename(columns={'측정일시':'date','측정소명':'district','이산화질소농도(ppm)':'no2',
                                  '오존농도(ppm)':'o3','이산화탄소농도(ppm)':'co2','아황산가스(ppm)':'so2',
                                  '미세먼지(㎍/㎥)':'pm10','초미세먼지(㎍/㎥)':'pm2_5'})
# 측정일시 -> date , 측정소명 -> district, 이산화질소 -> no2 , 오존 -> o3 , 이산화탄소 -> co2, 아황산가스 -> so2,
# 미세먼지 -> pm10 , 초미세먼지 -> pm2_5 롤 변경
df_mise_2020_copy

# 컬럼별 결측치 확인
df_mise_2020_copy.isnull().sum()

# 결측치 값 평균으로 대체 
df_mise_2020_copy.mean()
df_mise_2020_copy = df_mise_2020_copy.fillna(df_mise_2020_copy.mean())

# 컬럼별 결측치 다시 확인
df_mise_2020_copy.isnull().sum()

# 컬럼에 0값이 있는지 확인
zeroCol = []

for column in df_mise_2020_copy.columns:
    zeroCol.append(df_mise_2020_copy[df_mise_2020_copy[column] == 0]) # 컬럼값에 0이 있는지 확인

# pd.set_option('display.max_columns', 100) # 최대 행 출력!
df_mise_2020_copy

df_mise_2020_copy.describe() 
# 이산화질소농도에서 0값이 발견되었지만 충분히 0값이 있을 수 있음

# 연월일 date 컬럼을 이용하여 연월 date 컬럼만 추출
  
checkMonthArr = []
type(checkMonthArr)

# 날짜 데이터를 수치형에서 문자열로 변경
df_mise_2020_copy['date'] = df_mise_2020_copy['date'].astype(int).astype(str)
df_mise_2020_copy['date']
for i,idx in enumerate(df_mise_2020_copy.index):
    checkMonthArr.append(df_mise_2020_copy.loc[idx,'date'][0:6])
    if(i % 500 == 0):
        print('500건 수령 i =',i)
df_mise_2020_copy['checkMonth'] = checkMonthArr # df_mise_2020_copy에 checkMonth 변수 생성

df_mise_2020_copy

checkMonths = df_mise_2020_copy['checkMonth'].unique()
checkMonths      
"""
array(['202001', '202002', '202003', '202004', '202005', '202006',
       '202007', '202008', '202009', '202010', '202011', '202012'],
      dtype=object)
"""
del checkMonthArr

df_mise_2020_copy_01 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202001'] # 2020년도 1월 데이터만 추출
df_mise_2020_copy_02 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202002'] # 2020년도 2월 데이터만 추출
df_mise_2020_copy_03 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202003'] # 2020년도 3월 데이터만 추출
df_mise_2020_copy_04 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202004'] # 2020년도 4월 데이터만 추출
df_mise_2020_copy_05 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202005'] # 2020년도 5월 데이터만 추출
df_mise_2020_copy_06 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202006'] # 2020년도 6월 데이터만 추출
df_mise_2020_copy_07 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202007'] # 2020년도 7월 데이터만 추출
df_mise_2020_copy_08 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202008'] # 2020년도 8월 데이터만 추출
df_mise_2020_copy_09 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202009'] # 2020년도 9월 데이터만 추출
df_mise_2020_copy_10 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202010'] # 2020년도 10월 데이터만 추출
df_mise_2020_copy_11 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202011'] # 2020년도 11월 데이터만 추출
df_mise_2020_copy_12 = df_mise_2020_copy[df_mise_2020_copy['checkMonth'] == '202012'] # 2020년도 12월 데이터만 추출

df_mise_2020_copy_01
df_mise_2020_copy_12
# pd.options.display.max_rows = None # 필요시

# 연월 data 컬럼을 통해 계절별 data 컬럼 추출 
"""
3,4,5월 -> 봄
6,7,8월 -> 여름
9,10,11월 -> 가을
12,1,2 -> 겨울 로 출력
"""

df_mise_2020_copy['season'] = np.where(df_mise_2020_copy["checkMonth"].isin(['202003','202004','202005']) ,'spring',
np.where(df_mise_2020_copy["checkMonth"].isin(['202006','202007','202008']) ,'summer',
np.where(df_mise_2020_copy["checkMonth"].isin(['202009','202010','202011']) ,'autumn','winter')))
df_mise_2020_copy.season
df_mise_2020_copy

# 측정장소 data 컬럼을 통해 서울시 구별 data 컬럼 추출
# 총 50개의 측정장소를 서울시 25개의 구만 남겨두고 제거
districts = df_mise_2020_copy['district'].unique()
districts 
len(districts) # 50 

idx_district_no = df_mise_2020_copy[(df_mise_2020_copy['district'] == '공항대로') |
(df_mise_2020_copy['district'] == '관악산') | (df_mise_2020_copy['district'] == '남산') | (df_mise_2020_copy['district'] == '동작대로') | 
(df_mise_2020_copy['district'] == '북한산') | (df_mise_2020_copy['district'] == '서울숲') | (df_mise_2020_copy['district'] == '세곡') |
(df_mise_2020_copy['district'] == '시흥대로') | (df_mise_2020_copy['district'] == '신촌로') | (df_mise_2020_copy['district'] == '영등포로') |
(df_mise_2020_copy['district'] == '올림픽공원') | (df_mise_2020_copy['district'] == '자연사박물관') | (df_mise_2020_copy['district'] == '정릉로') |
(df_mise_2020_copy['district'] == '종로') | (df_mise_2020_copy['district'] == '천호대로') | (df_mise_2020_copy['district'] == '청계천로') |
(df_mise_2020_copy['district'] == '한강대로') | (df_mise_2020_copy['district'] == '행주') | (df_mise_2020_copy['district'] == '홍릉로') |
(df_mise_2020_copy['district'] == '화랑로') | (df_mise_2020_copy['district'] == '강남대로') | (df_mise_2020_copy['district'] == '강변북로') |
(df_mise_2020_copy['district'] == '도산대로') | (df_mise_2020_copy['district'] == '마포아트센터') | (df_mise_2020_copy['district'] == '궁동') ].index

df_mise_2020_copy = df_mise_2020_copy.drop(idx_district_no)
df_mise_2020_copy


df_mise_2020_copy.district
df_mise_2020_copy

########################################### 데이터 시각화 ############################################

# 측정일시별 미세먼지 시계열 그래프
meanGroupdate_pm10 = df_mise_2020_copy.groupby('date')['pm10'].mean()
meanGroupdate_pm10.plot()
plt.title('측정일시별 미세먼지 농도(㎍/㎥)')
plt.ylabel('미세먼지 농도(㎍/㎥)', fontsize=12)
plt.show() 

# 측정일시별 초미세먼지 시계열 그래프
meanGroupdate_pm2_5 = df_mise_2020_copy.groupby('date')['pm2_5'].mean()
meanGroupdate_pm2_5.plot()
plt.title('측정일시별 초미세먼지 농도(㎍/㎥)')
plt.ylabel('초미세먼지 농도(㎍/㎥)', fontsize=12)
plt.show() 

# 측정월별 미세먼지 시계열 그래프
meanGroupcheckMonth_pm10 = df_mise_2020_copy.groupby('checkMonth')['pm10'].mean()
meanGroupcheckMonth_pm10.plot()
plt.title('측정월별 미세먼지 농도(㎍/㎥)')
plt.ylabel('미세먼지 농도(㎍/㎥)',fontsize = 12)
plt.show()

# 측정월별 초미세먼지 시계열 그래프
meanGroupcheckMonth_pm2_5 = df_mise_2020_copy.groupby('checkMonth')['pm2_5'].mean()
meanGroupcheckMonth_pm2_5.plot()
plt.title('측정월별 초미세먼지 농도(㎍/㎥)')
plt.ylabel('초미세먼지 농도(㎍/㎥)',fontsize = 12)
plt.show()

##### seaborn 이용
### barplot
# 계절별 미세먼지 barplot 시각화
df_mise_2020_copy
sns.barplot(data = df_mise_2020_copy,x="season",y="pm10",order = ['spring','summer','autumn','winter'])
plt.title('계절별 미세먼지 barplot')
plt.show()

# 계절별 초미세먼지 barplot 시각화
sns.barplot(data = df_mise_2020_copy,x="season",y="pm2_5",order = ['spring','summer','autumn','winter'])
plt.title('계절별 초미세먼지 barplot')
plt.show()

# 서울시 구별 미세먼지 barplot
# 오름차순으로 정렬
meanGroupdistrict_pm10 = df_mise_2020_copy.groupby(['district'])['pm10'].mean().to_frame().sort_values(by='pm10')
plt.figure(figsize=(12,4))
sns.barplot(data=meanGroupdistrict_pm10, x=meanGroupdistrict_pm10.index, y='pm10',palette="Blues")
plt.title("서울시 구별 미세먼지 barplot")
plt.show()

# 내림차순으로 정렬
meanGroupdistrict_pm10 = df_mise_2020_copy.groupby(['district'])['pm10'].mean().to_frame().sort_values(by='pm10',ascending=False)
plt.figure(figsize=(12,4))
sns.barplot(data=meanGroupdistrict_pm10, x=meanGroupdistrict_pm10.index, y='pm10',palette="Blues_r")
plt.title("서울시 구별 미세먼지 barplot")
plt.show()

### boxplot
# 계절별 미세먼지 boxplot 시각화
sns.boxplot(
    data = df_mise_2020_copy,
    x = 'season',
    y = 'pm10',
    order = ['spring','summer','autumn','winter']
)
plt.title('계절별 미세먼지 boxplot')
plt.show()

# 서울시 구별 미세먼지 boxplot 시각화
sns.boxplot(
    data = df_mise_2020_copy,
    x = 'district',
    y = 'pm10'
)
plt.title('서울시 구별 미세먼지 boxplot')
plt.show()

###scatter

# 미세먼지, 초미세먼지 산점도로 그리기 
#  계절별 다른 색깔로 산점도 그리기
df_mise_2020_copy
len(df_mise_2020_copy['season'].unique())
categories = df_mise_2020_copy['season'].unique() # 중복 배제한 데이터 추출
categories

# 봄일 경우만 true이고, 다른 색깔로 산점도 그리기
tmp1 = df_mise_2020_copy.assign(isspring = df_mise_2020_copy.season=="spring") # isspring은 계절이 봄일 경우 True
tmp1.head()
sns.scatterplot(data=tmp1,x="pm10",y="pm2_5", hue = 'isspring' ) # hue를 통해 봄일경우만 다른 색깔로 확인 가능
plt.show()

# 여름일 경우만 true이고, 다른 색깔로 산점도 그리기
tmp2 = df_mise_2020_copy.assign(issummer = df_mise_2020_copy.season=="summer") # issummer은 계절이 여름일 경우 True
tmp2.head()
sns.scatterplot(data=tmp2,x="pm10",y="pm2_5", hue = 'issummer' ) # hue를 통해 여름일경우만 다른 색깔로 확인 가능
plt.show()

# 가을일 경우만 true이고, 다른 색깔로 산점도 그리기
tmp3 = df_mise_2020_copy.assign(isautumn = df_mise_2020_copy.season=="autumn") # isautumn은 계절이 가을일 경우 True
tmp3.head()
sns.scatterplot(data=tmp3,x="pm10",y="pm2_5", hue = 'isautumn' ) # hue를 통해 가을일경우만 다른 색깔로 확인 가능
plt.show()

# 겨울일 경우만 true이고, 다른 색깔로 산점도 그리기
tmp4 = df_mise_2020_copy.assign(iswinter = df_mise_2020_copy.season=="winter") # iswinter은 계절이 겨울일 경우 True
tmp4.head()
sns.scatterplot(data=tmp4,x="pm10",y="pm2_5", hue = 'iswinter' ) # hue를 통해 겨울일경우만 다른 색깔로 확인 가능
plt.show()

#####################################################################################################

"""
상관계수가 양의 값을 가질때 양의 상관관계를 갖는다고 할 수 있고,
상관계수가 음의 값을 가질때 음의 상관관계를 갖는다고 할 수 있다.
상관계수(r)의 기준은 아래와 같다.
r ≥ 0.8 일때, 강한 상관이 있고,
0.6 ≤ r < 0.8 일때, 상관이 있고,
0.4 ≤ r < 0.6 일때, 약한 상관이 있고,
r ≤ 0.4 일때, 거의 상관이 없다.
"""
### 각 변수간 산점도 그리기 
import matplotlib.pylab as pylab

# 미세먼지와 초미세먼지 
x = df_mise_2020_copy['pm10']
y = df_mise_2020_copy['pm2_5']
# 추세선을 위한 계산 - 1차원의 polynomial(다항식)을 계산하기 위한 코드입니다.
z = np.polyfit(x, y, 1) # (X,Y,차원) 정의
p = np.poly1d(z) # 1차원 다항식에 대한 연산을 캡슐화
# 그래프 그리기
pylab.plot(x,y,'o') #산점도를 뜻할 때 'o'라고 합니다.
pylab.plot(x,p(x),"r--")
pylab.title('미세먼지와 초미세먼지 scatterplot')
pylab.show()        # 강한 양의 상관관계를 보임  

# 이산화탄소와 초미세먼지 
x = df_mise_2020_copy['co2']
y = df_mise_2020_copy['pm2_5']
# 추세선을 위한 계산 - 1차원의 polynomial(다항식)을 계산하기 위한 코드입니다.
z = np.polyfit(x, y, 1) # (X,Y,차원) 정의
p = np.poly1d(z) # 1차원 다항식에 대한 연산을 캡슐화
# 그래프 그리기
pylab.plot(x,y,'o') #산점도를 뜻할 때 'o'라고 합니다.
pylab.plot(x,p(x),"r--")
pylab.title('이산화탄소와 초미세먼지 scatterplot')
pylab.show()        # 양의 상관관계를 보임 

# 이산화탄소와 미세먼지 
x = df_mise_2020_copy['co2']
y = df_mise_2020_copy['pm10']
# 추세선을 위한 계산 - 1차원의 polynomial(다항식)을 계산하기 위한 코드입니다.
z = np.polyfit(x, y, 1) # (X,Y,차원) 정의
p = np.poly1d(z) # 1차원 다항식에 대한 연산을 캡슐화
# 그래프 그리기
pylab.plot(x,y,'o') #산점도를 뜻할 때 'o'라고 합니다.
pylab.plot(x,p(x),"r--")
pylab.title('이산화탄소와 미세먼지 scatterplot')
pylab.show()        # 약한 양의 상관관계를 보임 

# 피어슨 상관계수 구하기
# 각 변수간 상관계수 구하기

# 미세먼지, 초미세먼지 간 상관계수
corr = lambda p : p['pm10'].corr(p['pm2_5'])
re = corr(df_mise_2020_copy)
print('미세먼지와 초미세먼지 상관계수:', re) # 미세먼지와 초미세먼지 상관계수: 0.8656215965677029

# 이산화탄소, 초미세먼지 간 상관계수
corr = lambda p : p['co2'].corr(p['pm2_5'])
re = corr(df_mise_2020_copy)
print('이산화탄소와 초미세먼지 상관계수:', re) # 이산화탄소와 초미세먼지 상관계수: 0.6892530226393224

# 이산화탄소, 미세먼지 간 상관계수
corr = lambda p : p['co2'].corr(p['pm10'])
re = corr(df_mise_2020_copy)
print('이산화탄소와 미세먼지 상관계수:', re) # 이산화탄소와 미세먼지 상관계수: 0.5913630635410302

# 각 변수간 상관관계 확인
df_mise_2020_copy
sns.heatmap(df_mise_2020_copy[['no2','o3','co2','so2','pm10','pm2_5']].corr(),annot=True)
plt.show()

"""
결과해석) heatmap을 확인한 결과, 미세먼지(pm10)과 초미세먼지(pm2_5)의 상관계수가 0.87으로 강한 상관관계가 있고,
이산화탄소(co2)과 초미세먼지(pm2_5)의 상관계수가 0.69로 상관이 있고, 
이산화탄소(co2)과 미세먼지(pm10)의 상관계수가 0.59으로 약한 상관관계가 있는 것을 알 수 있다.-
"""

# 선형 회귀 분석
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

reg = sm.OLS.from_formula("pm10 ~ pm2_5", df_mise_2020_copy).fit()
reg.summary()
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                   pm10   R-squared:                       0.749
Model:                            OLS   Adj. R-squared:                  0.749
Method:                 Least Squares   F-statistic:                 2.719e+04
Date:                Sat, 24 Jul 2021   Prob (F-statistic):               0.00
Time:                        18:14:58   Log-Likelihood:                -33006.
No. Observations:                9100   AIC:                         6.602e+04
Df Residuals:                    9098   BIC:                         6.603e+04
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      9.2921      0.185     50.359      0.000       8.930       9.654
pm2_5          1.2581      0.008    164.901      0.000       1.243       1.273
==============================================================================
Omnibus:                     5989.723   Durbin-Watson:                   0.477
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           115548.017
Skew:                           2.875   Prob(JB):                         0.00
Kurtosis:                      19.482   Cond. No.                         46.8
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
"""
결과해석) R-squared값이 0.749로, 이는 회귀모형의 설명력이 74.9%라는 뜻이다. 
회귀모형 F값이 매우 작고, 유의확률이 유의수준(0.05)보다 작으므로
회귀모형이 유의하다고 할 수 있다.
회귀식은 미세먼지(pm10) = 9.2921 + 1.2581 * 초미세먼지(pm2_5) 이고,
초미세먼지(pm2_5) 변수의 유의확률이 유의수준(0.05)보다 작기때문에 유의한 변수라고 할 수 있다.
"""
# 다중 선형 회귀분석
df_mise_2020_copy.iloc[:,:]
df_mise_2020_copy.iloc[:,[-10,-9,-8,-7,-5]] # 이산화질소 부터 초미세먼지 까지 추출 (미세먼지는 종속변수로 사용)

X = df_mise_2020_copy.iloc[:,[-10,-9,-8,-7,-5]].values # 독립변수
y = df_mise_2020_copy['pm10'].values # 종속변수

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=10) # train data와 test data로 구분(7:3)

lr = LinearRegression() # 단순회귀분석 모형 객체 생성

lr.fit(X_train,y_train) # train data를 가지고 모형 학습

r_square = lr.score(X_test, y_test) # 학습을 마친 모형에 test data를 적용하여 결정계수(r제곱) 계산
# train data로 회귀분석을 한 결과값
print('R-square : ', r_square) # R-square :  0.7673382542793407

print('X 변수의 기울기: ',lr.coef_) # 회귀식의 기울기
 # X 변수의 기울기:  [5.94040392e-01 1.65367428e+02 2.81658638e+00 1.41241409e+03 1.21386871e+00]

print('절편: ',lr.intercept_) # 회귀식의 절편
# 절편:  0.34953599697532667

"""
해석) 다중 선형회귀분석을 하기위해 미세먼지(pm10) 변수를 종속변수로 설정하고,
미세먼지(pm10)을 제외한 나머지 변수들을 독립변수로 설정한뒤,
train data와 test data를 7:3으로 구분하여 train data로 모형학습을 한 결과
R-square 값이 약 0.767으로 회귀모형의 설명력이 76.7%라는 뜻이다.
train data로 회귀분석을 한 회귀식은 미세먼지(pm10) = 0.35 + 0.59 * 이산환질소(no2) + 165.37 * 오존(o3) + 2.82 * 이산화탄소(co2)
                        + 1412.41 * 아황산가스(so2) + 1.21 * 초미세먼지(pm2_5) 
"""

# 최종적으로 훈련한 모델을 통해 테스트 데이터에 대해 예측을 수행 할 수 있음

y_hat = lr.predict(X_test)
plt.figure(figsize=(10,5))

plt.plot(y_test, label = 'y_test')
plt.plot(y_hat, label = 'y_hat' )
plt.legend(loc = 'upper center')
plt.show()

# 다중 공선성 확인
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

result1, result2 = dmatrices('pm10 ~ no2 + o3 + co2 + so2 + pm2_5' , df_mise_2020_copy, return_type = 'dataframe')
vif = pd.DataFrame()
vif["Vif Factor"] = [variance_inflation_factor(result2.values, i) for i in range(result2.shape[1])]
vif["features"] = result2.columns
print(vif) # vif가 10이하 이어야함
"""
   Vif Factor   features
0   27.777434  Intercept
1    1.019967        no2
2    1.268931         o3
3    2.475292        co2
4    1.340083        so2
5    2.068324      pm2_5
"""
"""
결과해석) 다중공선성이 10보다 크면 독립변수간 서로 강한 상관관계가 있다고 할 수 있지만, 10보다 큰 변수가 없기때문에 그대로 진행도됨
"""

# 다중선형회귀분석

import statsmodels.formula.api as sm
result = sm.ols(formula = 'pm10 ~ no2 + o3 + co2 + so2 + pm2_5', data = df_mise_2020_copy).fit()
result.summary()
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                   pm10   R-squared:                       0.762
Model:                            OLS   Adj. R-squared:                  0.762
Method:                 Least Squares   F-statistic:                     5813.
Date:                Sat, 24 Jul 2021   Prob (F-statistic):               0.00
Time:                        18:19:41   Log-Likelihood:                -32775.
No. Observations:                9100   AIC:                         6.556e+04
Df Residuals:                    9094   BIC:                         6.561e+04
Df Model:                           5
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.5484      0.490      1.119      0.263      -0.413       1.509
no2            0.8516      1.358      0.627      0.531      -1.810       3.513
o3           152.9794      8.248     18.547      0.000     136.811     169.148
co2            2.7431      0.806      3.405      0.001       1.164       4.322
so2         1515.9120    119.897     12.643      0.000    1280.887    1750.937
pm2_5          1.2036      0.011    112.488      0.000       1.183       1.225
==============================================================================
Omnibus:                     6137.731   Durbin-Watson:                   0.539
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           132701.025
Skew:                           2.935   Prob(JB):                         0.00
Kurtosis:                      20.763   Cond. No.                     3.12e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.12e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
"""
해석) R-square 값이 0.762로, 이는 회귀모형의 설명력이 76.2%라는 뜻이다.
회귀모형 F값이 매우 작고, 유의확률이 유의수준(0.05)보다 작으므로
회귀모형이 유의하다고 할 수 있다.
하지만 이산화질소(no2) 변수가 유의하지 않다고 나오기 때문에 이산화질소(no2)를 제거하고 다시한번 회귀식을 만들어 본다.
"""

result = sm.ols(formula = 'pm10 ~ o3 + co2 + so2 + pm2_5', data = df_mise_2020_copy).fit()
result.summary()

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                   pm10   R-squared:                       0.762
Model:                            OLS   Adj. R-squared:                  0.762
Method:                 Least Squares   F-statistic:                     7267.
Date:                Tue, 27 Jul 2021   Prob (F-statistic):               0.00
Time:                        15:53:04   Log-Likelihood:                -32775.
No. Observations:                9100   AIC:                         6.556e+04
Df Residuals:                    9095   BIC:                         6.560e+04
Df Model:                           4
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.5610      0.490      1.145      0.252      -0.399       1.521
o3           152.6467      8.231     18.546      0.000     136.513     168.781
co2            2.7503      0.806      3.414      0.001       1.171       4.329
so2         1518.0041    119.847     12.666      0.000    1283.078    1752.931
pm2_5          1.2039      0.011    112.652      0.000       1.183       1.225
==============================================================================
Omnibus:                     6136.481   Durbin-Watson:                   0.539
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           132615.819
Skew:                           2.934   Prob(JB):                         0.00
Kurtosis:                      20.757   Cond. No.                     3.12e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.12e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

"""
해석) R-square 값이 0.762로, 이는 회귀모형의 설명력이 76.2%라는 뜻이다.
회귀모형 F값이 매우 작고, 유의확률이 유의수준(0.05)보다 작으므로
회귀모형이 유의하다고 할 수 있다.
적합된 회귀식은 미세먼지(pm10) = 0.56  + 152.65 * 오존(o3) + 2.75 * 이산화탄소(co2) + 1518 * 아황산가스(so2) + 1.2 * 초미세먼지(pm2_5) 이고,
모든 변수들은 유의하다고 할 수 있다.
"""

print('미세먼지 예측: ',result.predict()) 

"""
미세먼지 예측:  [33.98467429 39.75956728 38.67801825 ... 24.02055787 24.91919603 20.81919673]
"""
"""
결과해석) 다중선형회귀식을 통해 미세먼지 수치를 예측할 수 있음
"""

### 다중 선형 회귀 분석 시각화

from sklearn import linear_model
X = df_mise_2020_copy.iloc[:,[-9,-8,-7,-5]].values # 독립변수
y = df_mise_2020_copy['pm10'].values # 종속변수

# 다중회귀분석 모델 설계
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(X = pd.DataFrame(X), y = y )
linear_regression_model_prediction = linear_regression_model.predict(X = pd.DataFrame(X))

# 다중회귀분석 실제값 / 예측값 시각화
fig = plt.figure(figsize = (12,4))
graph = fig.add_subplot(1,1,1)
graph.plot(y[:100], marker = 'o', color = 'blue', label = '실제값')
graph.plot(linear_regression_model_prediction[:100], marker ='^', color = 'red', label = '예측값' ) # 100개의 행만 시행
graph.set_title('다중회귀분석 예측 결과',size=30)
plt.xlabel('횟수',size = 20)
plt.ylabel('미세먼지',size = 20)
plt.legend(loc = 'best')
plt.show()

# 위 그래프의 정확도는 76.2%이다

#########################################################################################################

# 시계열 분석
# 시계열 분석 : 시계열 분석은 말그대로, 현시점까지의 데이터로 앞으로 어떤 패턴의 차트를 그릴지 예측하는 분석기법

from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df_mise_2020_copy['date'] = pd.to_datetime(df_mise_2020_copy.iloc[:]['date'], format="%Y%m%d") # 날자 형식 바꾸기

df_mise_2020_copy
mise_ts = df_mise_2020_copy[['date','pm10','district']]
mise_ts

## 동작구 미세먼지 시계열 분석
mise_ts_dongjak = mise_ts[mise_ts['district'] == '동작구']

mise_ts_dongjak['date'] = pd.DataFrame(mise_ts_dongjak['date'])
mise_ts_dongjak.index = mise_ts_dongjak['date']
mise_ts_dongjak.set_index('date',inplace=True)

mise_ts_dongjak.index
mise_ts_dongjak.plot()
plt.title("동작구 미세먼지 시계열 그래프")
plt.show()

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

mise_ts_dongjak = mise_ts_dongjak.iloc[:,:1] # district변수 제거

from pmdarima.arima import auto_arima

model_arima= auto_arima(mise_ts_dongjak,trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,suppress_warnings=True,stepwise=False,seasonal=False)
"""
ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=3074.710, Time=0.01 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=3057.248, Time=0.05 sec
 ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=2971.779, Time=0.10 sec
 ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=2971.442, Time=0.18 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=3071.433, Time=0.05 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=2984.592, Time=0.10 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=2971.541, Time=0.15 sec
 ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=2973.442, Time=0.28 sec
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=3038.297, Time=0.12 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=2971.745, Time=0.15 sec
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=2973.296, Time=0.24 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=2975.448, Time=0.26 sec
 ARIMA(3,1,0)(0,0,0)[0] intercept   : AIC=3031.234, Time=0.17 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=2973.640, Time=0.25 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=2974.899, Time=0.34 sec

Best model:  ARIMA(0,1,3)(0,0,0)[0] intercept
Total fit time: 2.472 seconds
"""
model_arima.fit(mise_ts_dongjak)
"""
ARIMA(order=(0, 1, 3), scoring_args={}, suppress_warnings=True)
"""
"""
결과해석) ARIMA 모델에서 AIC 값이 가장 낮은 값이 가장 좋은 모델인 것을 알 수있는데, 
ARIMA 모델에서 가장 괜찮은 모델을 찾은 결과, ARIMA(0,1,3) 모델이 가장 괜찮은 모델인 것을 알 수 있다.
따라서 ARIMA(0,1,3)을 가지고 ARIMA 모델링을 해볼 필요가 있다.
"""

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(mise_ts_dongjak, order=(0,1,3))

model_fit = model.fit(disp=0)

print(model_fit.summary())
"""
                             ARIMA Model Results
==============================================================================
Dep. Variable:                 D.pm10   No. Observations:                  363
Model:                 ARIMA(0, 1, 3)   Log Likelihood               -1480.721
Method:                       css-mle   S.D. of innovations             14.267
Date:                Sat, 24 Jul 2021   AIC                           2971.442
Time:                        18:26:55   BIC                           2990.914
Sample:                             1   HQIC                          2979.182

================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
const           -0.0240      0.072     -0.335      0.737      -0.164       0.116
ma.L1.D.pm10    -0.3835      0.052     -7.323      0.000      -0.486      -0.281
ma.L2.D.pm10    -0.4470      0.050     -8.931      0.000      -0.545      -0.349
ma.L3.D.pm10    -0.0783      0.051     -1.544      0.123      -0.178       0.021
                                    Roots
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
MA.1            1.0587           -0.0000j            1.0587           -0.0000
MA.2           -3.3824           -0.7857j            3.4725           -0.4637
MA.3           -3.3824           +0.7857j            3.4725            0.4637
-----------------------------------------------------------------------------
"""
"""
model의 summary를 보시면 중간에 coef값과 P>|z|값이 있습니다.
coef값은 0에서 가장 떨어졌을때 가장 이상적이며 p값은 0과 가장 가까울때가 가장 좋습니다.
그러므로 q 값이 2인 ma L2가 가장 정확도가 높습니다.
"""
# 패턴이 있는지 확인하기 위해 잔차 플롯 확인(일정한 평균과 분산을 가지고 있는지)
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
"""
잔차 오차는 평균이 거의 0에 가깝고 분산이 균일하여 괜찮아 보입니다
"""
# 동작구 미세먼지 실제데이터와 예측데이터 비교
model_fit.plot_predict(dynamic=False)
plt.title("동작구 미세먼지 실제데이터와 예측데이터 비교")
plt.show()

"""
model_fit.forecast(steps = 5)로 향후 5일의 가격을 예측하여 pred_y로 정의 한다.
mise_ts_dongjak[359:]로 해주어 mise_ts_dongjak의 마지막 5일을 test_y로 정의 한다.
모델의 예측한 상한값, 하한값을 pred_y_upper, pred_y_lower로 정의 한다.
정의한 모든 값을 비교하여 5일동안의 상승 경향 예측이 얼마나 맞는지 평가 해본다.
"""

mise_ts_dongjak
forecast_data = model_fit.forecast(steps=5)
mise_test_df = mise_ts_dongjak[359:]
mise_test_df
"""
            pm10
date
2020-12-27  64.0
2020-12-28  63.0
2020-12-29  68.0
2020-12-30  36.0
2020-12-31  30.0
"""

# 마지막 5일의 예측 데이터 (2020-12-27 ~ 2020-12-31)
pred_y = forecast_data[0].tolist()
# 실제 5일의 데이터 (2020-12-27 ~ 2020-12-31)
test_y = mise_test_df.pm10.values

# 마지막 5일의 예측 데이터 최소값, 최대값
pred_y_lower = []
pred_y_upper = []

for lower_upper in forecast_data[2]:
    lower = lower_upper[0]
    upper = lower_upper[1]
    pred_y_lower.append(lower)
    pred_y_upper.append(upper)

"""
그리고 다음 코드는 이를 그래프로 시각화 한 것이다.
파란색 그래프는 모델이 예상한 최고 가격, 즉 상한가의 그래프이다.
그리고 빨간색은 모델이 예측한 하한가 그래프이고, 초록색은 실제 5일간의 가격 그래프, 노란색은 모델이 예측한 가격 그래프를 나타낸 것이다.
"""
plt.plot(pred_y, color='gold',label='pred_y')
plt.plot(pred_y_lower, color='red',label='pred_y_lower')
plt.plot(pred_y_upper, color='blue',label='pred_y_upper')
plt.plot(test_y, color = 'green',label='test_y')
plt.legend(loc='best')
plt.show()

## 서초구 미세먼지 시계열 분석
mise_ts_seocho = mise_ts[mise_ts['district'] == '서초구']

mise_ts_seocho['date'] = pd.DataFrame(mise_ts_seocho['date'])
mise_ts_seocho.index = mise_ts_seocho['date']
mise_ts_seocho.set_index('date',inplace=True)

mise_ts_seocho.index
mise_ts_seocho.plot()
plt.title("서초구 미세먼지 시계열 그래프")
plt.show()

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

mise_ts_seocho = mise_ts_seocho.iloc[:,:1] # district변수 제거

from pmdarima.arima import auto_arima

model_arima= auto_arima(mise_ts_seocho,trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,suppress_warnings=True,stepwise=False,seasonal=False)
"""
ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=3178.200, Time=0.01 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=3163.659, Time=0.04 sec
 ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=3077.149, Time=0.08 sec
 ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=3077.558, Time=0.16 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=3175.296, Time=0.08 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=3090.583, Time=0.13 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=3077.411, Time=0.14 sec
 ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=3079.411, Time=0.26 sec
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=3145.002, Time=0.12 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=3077.638, Time=0.14 sec
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=3079.409, Time=0.36 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=3081.021, Time=0.77 sec
 ARIMA(3,1,0)(0,0,0)[0] intercept   : AIC=3137.976, Time=0.13 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=3079.637, Time=0.31 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=3079.677, Time=0.29 sec

Best model:  ARIMA(0,1,2)(0,0,0)[0] intercept
Total fit time: 3.041 seconds
"""
model_arima.fit(mise_ts_seocho)
"""
ARIMA(order=(0, 1, 2), scoring_args={}, suppress_warnings=True)
"""
"""
결과해석) ARIMA 모델에서 AIC 값이 가장 낮은 값이 가장 좋은 모델인 것을 알 수있는데, 
ARIMA 모델에서 가장 괜찮은 모델을 찾은 결과, ARIMA(0,1,2) 모델이 가장 괜찮은 모델인 것을 알 수 있다.
따라서 ARIMA(0,1,2)을 가지고 ARIMA 모델링을 해볼 필요가 있다.
"""

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(mise_ts_seocho, order=(0,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
"""
                              ARIMA Model Results
==============================================================================
Dep. Variable:                 D.pm10   No. Observations:                  363
Model:                 ARIMA(0, 1, 2)   Log Likelihood               -1534.575
Method:                       css-mle   S.D. of innovations             16.550
Date:                Mon, 26 Jul 2021   AIC                           3077.149
Time:                        09:55:23   BIC                           3092.727
Sample:                             1   HQIC                          3083.341

================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
const           -0.0115      0.094     -0.123      0.902      -0.195       0.172
ma.L1.D.pm10    -0.4033      0.044     -9.088      0.000      -0.490      -0.316
ma.L2.D.pm10    -0.4928      0.044    -11.197      0.000      -0.579      -0.407
                                    Roots
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
MA.1            1.0729           +0.0000j            1.0729            0.0000
MA.2           -1.8912           +0.0000j            1.8912            0.5000
-----------------------------------------------------------------------------
"""
"""
model의 summary를 보시면 중간에 coef값과 P>|z|값이 있습니다.
coef값은 0에서 가장 떨어졌을때 가장 이상적이며 p값은 0과 가장 가까울때가 가장 좋습니다.
그러므로 q 값이 2인 ma L2가 가장 정확도가 높습니다.
"""
# 패턴이 있는지 확인하기 위해 잔차 플롯 확인(일정한 평균과 분산을 가지고 있는지)
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
"""
잔차 오차는 평균이 거의 0에 가깝고 분산이 균일하여 괜찮아 보입니다
"""
# 동작구 미세먼지 실제데이터와 예측데이터 비교
model_fit.plot_predict(dynamic=False)
plt.title("서초구 미세먼지 실제데이터와 예측데이터 비교")
plt.show()

"""
model_fit.forecast(steps = 5)로 향후 5일의 가격을 예측하여 pred_y로 정의 한다.
mise_ts_seocho[359:]로 해주어 mise_ts_seocho의 마지막 5일을 test_y로 정의 한다.
모델의 예측한 상한값, 하한값을 pred_y_upper, pred_y_lower로 정의 한다.
정의한 모든 값을 비교하여 5일동안의 상승 경향 예측이 얼마나 맞는지 평가 해본다.
"""

mise_ts_seocho
forecast_data = model_fit.forecast(steps=5)
mise_test_df = mise_ts_seocho[359:]
mise_test_df
"""
            pm10
date
2020-12-27  69.0
2020-12-28  68.0
2020-12-29  76.0
2020-12-30  38.0
2020-12-31  25.0
"""

# 마지막 5일의 예측 데이터 (2020-12-27 ~ 2020-12-31)
pred_y = forecast_data[0].tolist()
# 실제 5일의 데이터 (2020-12-27 ~ 2020-12-31)
test_y = mise_test_df.pm10.values

# 마지막 5일의 예측 데이터 최소값, 최대값
pred_y_lower = []
pred_y_upper = []

for lower_upper in forecast_data[2]:
    lower = lower_upper[0]
    upper = lower_upper[1]
    pred_y_lower.append(lower)
    pred_y_upper.append(upper)

"""
그리고 다음 코드는 이를 그래프로 시각화 한 것이다.
파란색 그래프는 모델이 예상한 최고 가격, 즉 상한가의 그래프이다.
그리고 빨간색은 모델이 예측한 하한가 그래프이고, 초록색은 실제 5일간의 가격 그래프, 노란색은 모델이 예측한 가격 그래프를 나타낸 것이다.
"""
plt.plot(pred_y, color='gold',label='pred_y')
plt.plot(pred_y_lower, color='red',label='pred_y_lower')
plt.plot(pred_y_upper, color='blue',label='pred_y_upper')
plt.plot(test_y, color = 'green',label='test_y')
plt.legend(loc='best')
plt.show()












