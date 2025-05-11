import json
import urllib3
import numba
import numpy as np
from datetime import datetime
import certifi
import re



area_dic = {'北海道/釧路':'014100',
            '北海道/旭川':'012000',
            '北海道/札幌':'016000',
            '青森県':'020000',
            '岩手県':'030000',
            '宮城県':'040000',
            '秋田県':'050000',
            '山形県':'060000',
            '福島県':'070000',
            '茨城県':'080000',
            '栃木県':'090000',
            '群馬県':'100000',
            '埼玉県':'110000',
            '千葉県':'120000',
            '東京都':'130000',
            '神奈川県':'140000',
            '新潟県':'150000',
            '富山県':'160000',
            '石川県':'170000',
            '福井県':'180000',
            '山梨県':'190000',
            '長野県':'200000',
            '岐阜県':'210000',
            '静岡県':'220000',
            '愛知県':'230000',
            '三重県':'240000',
            '滋賀県':'250000',
            '京都府':'260000',
            '大阪府':'270000',
            '兵庫県':'280000',
            '奈良県':'290000',
            '和歌山県':'300000',
            '鳥取県':'310000',
            '島根県':'320000',
            '岡山県':'330000',
            '広島県':'340000',
            '山口県':'350000',
            '徳島県':'360000',
            '香川県':'370000',
            '愛媛県':'380000',
            '高知県':'390000',
            '福岡県':'400000',
            '佐賀県':'410000',
            '長崎県':'420000',
            '熊本県':'430000',
            '大分県':'440000',
            '宮崎県':'450000',
            '鹿児島県':'460100',
            '沖縄県/那覇':'471000',
            '沖縄県/石垣':'474000'
            }

area_kanji_to_romaji = {
    '北海道内市町村': {'region_id': (22), 'region_name': 'hokkaido', 'region_code': (79)},
    ('青森県内市町村', '岩手県内市町村', '宮城県内市町村', 
     '秋田県内市町村', '山形県内市町村', '福島県内市町村'): 
        {'region_id': (23, 24, 25, 26, 27, 28), 'region_name': 'touhoku', 'region_code': (80)},
    ('茨城県内市町村', '栃木県内市町', '群馬県内市町村', 
     '埼玉県内市町村', '千葉県内市町村', '東京都内市町村', '神奈川県内市町村'): 
        {'region_id': (29, 30, 31, 32, 33, 34, 35), 'region_name': 'kantou', 'region_code': (81)},
    ('新潟県内市町村', '富山県内市町村', '石川県内市町', '福井県内市町', 
     '山梨県内市町村', '長野県内市町村', '岐阜県内市町村', '静岡県内市町', '愛知県内市町村'): 
        {'region_id': (36, 37, 38, 39, 40, 41, 42, 43, 44), 'region_name': 'tyuubu', 'region_code': (82)},
    ('三重県内市町', '滋賀県内市町', '京都府内市町村', 
     '大阪府内市町村', '兵庫県内市町', '奈良県内市町村', '和歌山県内市町村'): 
        {'region_id': (45, 46, 47, 48, 49, 50, 51), 'region_name': 'kinki', 'region_code': (83)},
    ('鳥取県内市町村', '島根県内市町村', '岡山県内市町村', '広島県内市町', '山口県内市町'): 
        {'region_id': (52, 53, 54, 55, 56), 'region_name': 'chugoku', 'region_code': (84)},
    ('徳島県内市町村', '香川県内市町', '愛媛県内市町', '高知県内市町村'): 
        {'region_id': (57, 58, 59, 60), 'region_name': 'shikoku', 'region_code': (85)},
    ('福岡県内市町村', '佐賀県内市町', '長崎県内市町', '熊本県内市町村', 
     '大分県内市町村', '宮崎県内市町村', '鹿児島県内市町村', '沖縄県内市町村'): 
        {'region_id': (61, 62, 63, 64, 65, 66, 67, 68), 'region_name': 'kyuusyuu', 'region_code': (87)}
}


class City:
    def __init__(self, prefecture):
        self.prefecture = prefecture 


    def get_municipalities(self, block_area, prefecture, region_id, region_code):
        """都道府県名から市町村のリストを取得する関数"""
        region_id_number = region_id[0]
        region_code_number = region_code[0]
        region_name_char = block_area[0]
        url = fr"https://www.j-lis.go.jp/spd/code-address/{region_name_char}/cms_1{region_id_number}141{region_code_number}.html"
        # urllib3のHTTPプールマネージャを作成
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            ca_certs=certifi.where()
        )
        # urllib3を使用してデータを取得
        response = http.request('GET', url).data.decode('utf-8')
    
        # 都道府県名が含まれる部分を検索
        prefecture_pattern = re.compile(r'(<h1>.*?{}.*?</h1>)'.format(re.escape(prefecture)))  # 都道府県名を含むタグを検索
        prefecture_match = prefecture_pattern.search(response)
    
        if prefecture_match:
            # 都道府県が見つかったら、その部分を無視して以降の市町村名を抽出
            prefecture_section = response[prefecture_match.end():]
            municipalities = re.findall(r'([一-龯]+(?:市|町|村|区|島))', prefecture_section)
            exclude_list = ["都道府県別市区町村", "全国町村", "東京都千代田区一番町"]
            municipalities = [m for m in municipalities if m not in exclude_list]
    
            return list(set(municipalities))
        else:
            # 見つからなかった場合は空のリストを返す
            return []

    
    def get_region_info(self, prefecture_name):
        """都道府県名からregion_idとregion_codeを取得する関数"""
        region_ids = []
        region_codes = []
        region_names = []
    
        for key, value in area_kanji_to_romaji.items():
            if isinstance(key, tuple):
                if prefecture_name in key:
                    index = key.index(prefecture_name)
                    region_ids.append(value['region_id'][index])
                    region_codes.append(value['region_code'])
                    region_names.append(value['region_name'])
            elif prefecture_name == key:
                region_ids.append(value['region_id'])
                region_codes.append(value['region_code'])
                region_names.append(value['region_name'])
        
        if region_ids and region_codes and region_names:
            return region_ids, region_codes, region_names
        else:
            return None, None, None
    
    
    def transform_prefecture(self, prefecture):
        if '県' in prefecture:
            if any(substring in prefecture for substring in ['栃木', '石川', '福井', '静岡', '三重', '滋賀', '広島', '山口', '香川', '愛媛', '佐賀', '長崎']):
                return prefecture.replace('県', '県内市町')
            else:
                return prefecture.replace('県', '県内市町村')
        elif '道' in prefecture:
            return prefecture.replace('道', '道内市町村')
        elif '府' in prefecture:
            return prefecture.replace('府', '府内市町村')
        elif '都' in prefecture:
            return prefecture.replace('都', '都内市町村')
        else:
            return prefecture


class Analysis:
    def __init__(self, selected_area, city):
        self.selected_area = selected_area
        self.city = city
 

    def run(self):
        selected_area_code = area_dic.get(self.selected_area)
        if selected_area_code is not None:
            jma_data = self.get_jma_data(selected_area_code)
            if jma_data:
                result_text = self.generate_result_text(jma_data)
                print(result_text)
            else:
                print("気象データの取得に失敗しました。")
        else:
            print("選択した都道府県はサポートされていません。")


    def get_jma_data(self, area_code):
        jma_url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"

        # urllib3のHTTPプールマネージャを作成
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            ca_certs=certifi.where()
        )

        try:
            # urllib3を使用してデータを取得
            response = http.request('GET', jma_url)
            jma_data = json.loads(response.data.decode('utf-8'))
            return jma_data
        except Exception as e:
            print(f"データの取得中にエラーが発生しました: {e}")
            return


    # 2D配列を生成する関数
    def create_2d_array(self, json_data):
        # 各地域のweatherCodesを取得
        names = []
        weather_codes = []
        date = []
        for entry in json_data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                dates = time_data.get("timeDefines", [])
                date.append(dates)
                areas = time_data.get("areas", [])
                for i, area in enumerate(areas):
                    area_weather_codes = area.get("weatherCodes", [])
                    name = area["area"]["name"]
                    weather_codes.append(area_weather_codes)
                    names.append(name)

        # 最大の地域数を取得
        max_area_count = max(len(area_codes) for area_codes in weather_codes)

        # 2D配列を初期化
        result = np.empty((len(weather_codes), max_area_count), dtype=np.float32)

        # weatherCodesを2D配列にセット
        for i, area_codes in enumerate(weather_codes):
            result[i, :len(area_codes)] = area_codes

        return result, names, date


    def generate_result_text(self, jma_data):

        city = City(self.selected_area)
        # 入力値の取得
        selected_area = self.selected_area
        # 県、府、道、都を取り除く
        prefecture = selected_area.split('/')[0]  # /以降を省略
        # # area_kanji_to_romajiから自動的に判別
        # for key, value in area_kanji_to_romaji.items():
        #     if isinstance(key, tuple):
        #         if any(prefecture in k for k in key):
        #             romaji_area = value
        #             break
        #     elif prefecture == key:
        #         romaji_area = value
        #         break
        # else:
        #     raise KeyError(f"'{prefecture}' is not found in area_kanji_to_romaji")
        
        transformed_prefecture = city.transform_prefecture(prefecture)
        region_id, region_code, region_name = city.get_region_info(transformed_prefecture)
        
        selected_city = city.get_municipalities(region_name, transformed_prefecture, region_id, region_code)

        result_text = f"気象庁データ: {selected_area}\n"
        result_text += f"今日の天気: {jma_data[0]['timeSeries'][0]['areas'][0]['weathers'][0]}\n"
        result_text += f"今日の風: {jma_data[0]['timeSeries'][0]['areas'][0]['winds'][0]}\n"


        weather_matrix, names, dates = self.create_2d_array(jma_data)

        date_order_count = 0

        names = list(set(names))

        for date in dates:
            if date_order_count == 4:
                continue
            date_order_count += 1
            for city in selected_city:
                if re.fullmatch(r'.*島.*', city):
                    continue
                day_count = 0
                if datetime.fromisoformat(date[day_count]).hour == 0 and len(date) == 7:
                    result_text += f"----------------------------------[地域, 一週間ごと]---------------------------------------\n"
                    day_count += 1
                else:
                    result_text += f"----------------------------------[地域, 時間ごと]-----------------------------------------\n"
                    day_count += 1
                for i in range(len(date)):
                    # ISO 8601形式の日付と時刻を解析してPythonのdatetimeオブジェクトに変換
                    input_datetime = datetime.fromisoformat(date[i])
                    weekly = input_datetime.weekday()
                    # 24時間制のフォーマットで日付と時刻を文字列に変換
                    format_month = str(int(str(input_datetime.month))) + "月"
                    format_hour = str(int(str(input_datetime.hour))) + "時"
                    format_time = str(int(str(input_datetime.minute))) + "分"
                    if weekly == 0:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　月曜日")
                    elif weekly == 1:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　火曜日")
                    elif weekly == 2:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　水曜日")
                    elif weekly == 3:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　木曜日")
                    elif weekly == 4:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　金曜日")
                    elif weekly == 5:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　土曜日")
                    else:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　日曜日")
                    result_text += f"日付: {formatted_datetime}\n"
                    result_text += f"地域名: {city}\n"

                    # 天気コードからなる2D配列を作成
                    code = weather_matrix
                    results = np.array([[float(code) for code in row] for row in code], dtype=np.float32)

                    result = self.cuda_ridge_detection(results, 0.5)  # 調整後の闘値を使用して再度リッジ検出を実行

                    # 日時系列ごとにトータルリッジ検出結果と闘値を計算
                    total_ridge, threshold = self.calculate_total_ridge_and_threshold(result , i)

                    # 他の条件に応じて闘値を調整
                    if total_ridge >= 10:
                        threshold += 0.5  # トータルリッジ検出値が10以上の場合、闘値を0.5増加させる

                    # 降水確率が高い場合に闘値を下げる
                    jma_rainfalls = self.get_precipitation_probability(jma_data)

                    # トータルリッジが2以上の場合
                    if total_ridge >= 2:
                        # 降水確率を50%以上に設定
                        jma_rainfalls[i] = min(max(total_ridge * 10, 50), 100)
                    # トータルリッジが0の場合
                    elif total_ridge == 0:
                        # 降水確率を0%に設定
                        jma_rainfalls[i] = 0
                    # トータルリッジが1未満の場合
                    else:
                        # 降水確率をトータルリッジの値に応じて設定
                        jma_rainfalls[i] = min(total_ridge * 10, 100)


                    # total_ridge = np.sum(result[0])

                    # 降水確率と平均風速を取得

                    if self.get_winds(jma_data) is not None:
                       winds, wind_speeds = self.get_winds(jma_data)
                    else:
                       winds, wind_speeds = None, None


                    # 天候予測
                    low_temperature, up_temprature = self.calculate_average_temperature(jma_data)
                    average_rainfalls = self.calculate_average_rainfall(jma_rainfalls)
                    snow_predicted, predicted_weather, snow_probability = self.predict_weather(
                        1.0,
                        5.0,
                        10.0,
                        low_temperature,
                        up_temprature,
                        average_rainfalls,
                        total_ridge,
                        jma_rainfalls,
                        wind_speeds,
                        i
                    )

                    # 修正：snow_probabilityを考慮して雪の確率を上げる
                    if snow_probability > 0.1:
                       snow_probability += 0.2  # 雪の確率が一定値以上ならばさらに上げる（適宜調整）
                    elif snow_predicted:
                       snow_probability += 0.2

                    result_text += f"降水確率: {jma_rainfalls[i]}%\n"
                    # Safe index access with validation
                    if winds and i < len(winds):
                       result_text += f"風: {winds[i]}\n"
                    else:
                       result_text += "風: データなし\n"

                    result_text += f"天気予測：{predicted_weather}\n"

        return result_text


    def calculate_total_ridge_and_threshold(self, results, index):
    # ここで日時系列ごとにトータルリッジ検出結果と闘値を計算
        total_ridge = np.sum(results[0][index])
        mean_value = np.mean(results[0][index])
        std_deviation = np.std(results[0][index])
        threshold = (mean_value + 2 * std_deviation) / 10 ** 34  # 例: (平均値 + 2倍の標準偏差) / 10の34乗で少数点数にする
        return total_ridge, threshold


    def get_precipitation_probability(self, data):
        pops = []
        for entry in data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                areas = time_data.get("areas", [])
                for area in areas:
                    pops.extend(area.get("pops", []))
        # 空文字列や空の要素を取り除く
        pops = [value for value in pops if value]
        return pops


    def calculate_average_rainfall(self, jma_rainfalls):
        pops = []  # リストとして初期化
        if jma_rainfalls:
            for entry in jma_rainfalls:
                try:
                    # 各要素を数値に変換してリストに追加
                    pops.append(float(entry))
                except (ValueError, TypeError):
                    pops = self.get_precipitation_probability(jma_rainfalls)
        # 空文字列や空の要素を取り除く
        pops = [value for value in pops if value]
        # リストに要素があれば平均を計算
        return np.mean(pops, dtype=np.float32) if pops else None


    def calculate_average_temperature(self, data):
        temperatures = []
        lower_temperatures = []
        upper_temperatures = []
        for entry in data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                areas = time_data.get("areas", [])
                for area in areas:
                    temperatures.extend(area.get("temps", []))
                    lower_temperatures.extend(area.get("tempsMinLower", []))
                    upper_temperatures.extend(area.get("tempsMaxLower", []))
        # 空文字列や空の要素を取り除く
        low_temperatures = [float(value) for value in lower_temperatures if value]
        up_temperatures = [float(value) for value in upper_temperatures if value]
        for i in range(len(temperatures)):
            if i % 2 == 0:
                low_temperatures.insert(0, float(temperatures[i]))
            else:
                up_temperatures.insert(0, float(temperatures[i]))
        return low_temperatures, up_temperatures


    def cuda_ridge_detection(self, data, thres):
        rows, cols = data.shape
        count = np.zeros_like(data, dtype=np.float32)
        for i in numba.prange(1, rows - 1):
            for j in range(1, cols - 1):
                if (
                    i > 0
                    and j > 0
                    and i < (rows - 1)
                    and j < (cols - 1)
                    and data[i, j] > thres
                    and not np.isnan(data[i, j])
                ):
                    step_i = i
                    step_j = j
                    for k in range(1000):
                        if (
                            step_i == 0
                            or step_j == 0
                            or step_i == (rows - 1)
                            or step_j == (cols - 1)
                        ):
                            break
                        index = 4
                        vmax = -np.inf
                        for ii in range(3):
                            for jj in range(3):
                                value = data[step_i + ii - 1, step_j + jj - 1]
                                if value > vmax:
                                    vmax = value
                                    index = jj + 3 * ii
                        if index == 4 or vmax == data[step_i, step_j] or np.isnan(vmax):
                            break
                        row = index // 3
                        col = index % 3
                        count[step_i - 1 + row, step_j - 1 + col] += 1
                        step_i, step_j = step_i - 1 + row, step_j - 1 + col

        # weather_array_normalizedの処理
        weather_normalized = np.mean(data)

        # 平均値が特定の閾値を超えるかどうかの判定
        threshold_exceeded = weather_normalized > 0.5

        # 閾値の超過判定結果を返す
        return count, threshold_exceeded


    def get_winds(self, data):
        winds = []
        for entry in data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                areas = time_data.get("areas", [])
                for area in areas:
                    winds.extend(area.get("waves", []))
        # 空文字列や空の要素を取り除く
        winds = [value for value in winds if value]
        try:
           if winds:
                   wind_speeds = []
                   for wind in winds:
                       if wind is not None:
                           # 風速の文字列から数字と単位（メートル）を取り除いて浮動小数点数に変換
                           for wind in winds:
                               wind_values = re.findall(r'[\d.]+', wind)  # 風速値を正規表現で抽出
                               for wind_speed in wind_values:
                                   wind_speeds.append(float(wind_speed))  # 風速を浮動小数点数に変換してリストに追加
                                   return winds, wind_speeds
        except Exception as e:
            print(f"風速情報の抽出中にエラーが発生しました: {e}")
        return None


    def predict_weather(self, low_temperature_threshold, up_temperature_threshold, precipitation_threshold, low_average_temperature, up_average_temperature, average_rainfall, total_ridge, jma_rainfalls, winds, i):

        # 降水確率を取得
        precipitation_probability = float(jma_rainfalls[i])

        snow_probability = precipitation_probability

        # 平均気温の閾値を調整
        if low_average_temperature[i] <= -2.0:
            snow_probability += 0.3  # 平均気温が-2.0度以下の場合、雪の確率を増加

        # 降水量が影響する条件を調整
        if average_rainfall >= 5.0:
            snow_probability += 0.2  # 平均降水量が5.0mm以上の場合、雪の確率を増加

        # 雪の予測ロジックを調整
        if (
            low_average_temperature[i] <= low_temperature_threshold
            and up_average_temperature[i] <= up_temperature_threshold
            and float(precipitation_probability) >= precipitation_threshold
            and 10 <= snow_probability <= 30
        ):
            snow_predicted = True
        else:
            snow_predicted = False
            snow_probability = 0.0

        # 天気予測のロジック
        if snow_predicted:
            predicted_weather = "雪"
        else:
            if total_ridge == 0 or precipitation_probability <= 20:
                predicted_weather = "晴れ"
                snow_probability = 0.0
            elif total_ridge == 1 or 20 <= precipitation_probability <= 40:
                predicted_weather = "曇り"
                snow_probability = 0.0
            elif total_ridge >= 2 or precipitation_probability >= 40:
                predicted_weather = "雨"
                snow_probability = 0.0

        # 修正：snow_probabilityも返す
        return snow_predicted, predicted_weather, snow_probability



def main():
    selected_area = input("都道府県を入力してください：")
    city = City(selected_area)
    analysis = Analysis(selected_area, city)

    analysis.run()



if __name__ == "__main__":
    main()