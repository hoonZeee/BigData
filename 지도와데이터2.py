
#%%
import folium
import geokakao as gk
import webbrowser

loc=gk.convert_address_to_coordinates('서울 종로구 사직로 161')

map = folium.Map(location=loc,zoom_start=16)
folium.Marker(location=loc, popup='경북궁',
              icon=folium.Icon(icon='flag',color='red')).add_to(map)


file_path='/Users/ijihun/bigData/Population_SIG/map.html'

map.save(file_path)

# 기본 웹브라우저로 파일 열기
webbrowser.open('file://' + file_path)


#%%
html_start = html = '<div \
style="\
font-size: 12px;\
color: blue;\
background-color:rgba(255, 255, 255, 0.2);\
width:85px;\
text-align:left;\
margin:0px;\
"><b>'
html_end = '</b></div>'

folium.Marker(location=loc,
              icon=folium.DivIcon(
                  icon_anchor=(0, 0),  # 텍스트 위치 설정
                  html=html_start+'경북궁'+html_end
              )).add_to(map)

file_path='/Users/ijihun/bigData/Population_SIG/map.html'
map.save(file_path)

# 기본 웹브라우저로 파일 열기
webbrowser.open('file://' + file_path)
#%%
import folium
import json
import pandas as pd
m=folium.Map(location=[35.1796,129.0756],zoom_start=12)

def popup_function(feature):
    return folium.Popup(feature['properties']['SIB_KOR_NM'])

districts=[
    {"name":"중구","lat":35.1065,"lon":129.0323,"population":46208},
    {"name":"서구","lat":35.0824,"lon":129.0206,"population":109742},
    {"name":"동구","lat":35.1293,"lon":129.0450,"population":93417},
    {"name":"영도구","lat":35.0911,"lon":129.0680,"population":114673},
    {"name":"부산진구","lat":35.1634,"lon":129.0533,"population":363678},
    {"name":"남구","lat":35.1367,"lon":129.0842,"population":281959},
    {"name":"해운대구","lat":35.1631,"lon":129.1636,"population":423660},
]
    
population_data={d["name"]:d["population"] for d in districts}

geo_busan=json.load(open('/Users/ijihun/bigData/Population_SIG/sigoo.geojson',encoding='UTF-8'))
feature=[x for x in geo_busan['features'] if x['properties']['SIG_CD'].startswith('26')]
geo_busan['features']=feature


df_pop=pd.read_csv('/Users/ijihun/bigData/Population_SIG//Population_SIG.csv')
df_pop.info()

df_pop['code']=df_pop['code'].astype(str)
pcode_df=df_pop.query('code.str.startswith("26")')



pusan_pop=pd.read_csv('/Users/ijihun/bigData/Population_SIG/pusan_pop(1).csv',encoding='euc-kr')

pcode_df.drop(['pop'],axis=1,inplace=True)

pusan_pop['총인구']=pusan_pop['총인구'].astype(int)
pcode=pd.merge(pcode_df, pusan_pop,how='inner',on='region')
#%%

html_start = html = '<div \
style="\
font-size: 20px;\
color: red;\
background-color:rgba(255, 255, 255, 0);\
width:85px;\
text-align:left;\
margin:0px;\
"><b>'
html_end = '</b></div>'
bins=list(pcode['총인구'].quantile([0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1]))

map_pusan =folium.Map(location=[35.17,129.07],             
           zoom_start=12,
           tiles='cartodbpositron')
folium.Choropleth(
    geo_data=geo_busan,
    data=pcode,
    columns=('code','총인구'),
    key_on='feature.properties.SIG_CD',
    fill_color='YlGnBu',
    legend_name="부산 구/군별 인구",
    nan_fill_color='White', # 결측치 색상
    fill_opacity=0.8,
    line_opacity=0.5, 
    bins=bins).add_to(map_pusan)

for district in districts:   

    folium.CircleMarker(
        location=[district['lat'],district['lon']],
        radius= district['population']/10000,               
        color='blue', fill=True,fill_color='blues',
        fill_opacity=0.6,
        popup=folium.Popup(f'{district["name"]} 인구: {district["population"]} 명',
                          parse_html=True )).add_to(map_pusan)
    
for district in districts:
    
    folium.Marker(location=[district['lat'],district['lon']],
                  icon=folium.DivIcon(
                      icon_anchor=(0, 0),  # 텍스트 위치 설정
                     html=f"""
                   <div style="font-size: 12px; color: red; font-size: 20px; white-space: nowrap;">
                   {district["name"]}
                   </div>
                    """
                  )).add_to(map_pusan)
    
    
    
        
    
    
file_path='/Users/ijihun/bigData/Population_SIG/map.html'
map.save(file_path)
map_pusan.save(file_path)
webbrowser.open('file://' + file_path)
