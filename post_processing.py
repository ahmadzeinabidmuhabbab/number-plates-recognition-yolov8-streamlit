import re

def huruf_sebelum_angka(string):
    # Cari pola huruf sebelum angka dengan regex
    match = re.match(r'([a-zA-Z]*)(\d*)', string)
    if match:
        return match.group(1)
    else:
        return None

def huruf_setelah_angka(string):
    # Cari pola huruf setelah angka dengan regex
    match = re.search(r'\d+([A-Z]+.*)', string, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

def semua_angka(string):
    # Menggunakan regex untuk menemukan semua digit dalam string
    angka = re.findall(r'\d', string)
    if angka:
        return ''.join(angka)  # Menggabungkan semua digit yang ditemukan menjadi satu string
    else:
        return None

def hanya_huruf(string):
    if string is None:
        return None
    # Menggunakan regex untuk menemukan semua huruf dalam string
    huruf = re.findall(r'[A-Za-z]', string)
    if huruf:
        return ''.join(huruf)  # Menggabungkan semua huruf yang ditemukan menjadi satu string
    else:
        return None

def transform_df(df):
    df['len_label'] = df['Labels'].apply(lambda x: len(x) if x is not None else 0)
    df['huruf_sebelum_angka'] = df['Labels'].apply(huruf_sebelum_angka)
    df['huruf_sebelum_angka'] = df['huruf_sebelum_angka'].apply(hanya_huruf)
    df['huruf_setelah_angka'] = df['Labels'].apply(huruf_setelah_angka)
    df['huruf_setelah_angka'] = df['huruf_setelah_angka'].apply(hanya_huruf)
    df['semua_angka'] = df['Labels'].apply(semua_angka)
    df['len_huruf_sebelum_angka'] = df['huruf_sebelum_angka'].apply(lambda x: len(x) if x is not None else 0)
    df['len_huruf_setelah_angka'] = df['huruf_setelah_angka'].apply(lambda x: len(x) if x is not None else 0)
    df['len_semua_angka'] = df['semua_angka'].apply(lambda x: len(x) if x is not None else 0)
    return df

list_huruf_sebelum_angka = ['A', 'B', 'AB', 'AD', 'AE', 'AA', 'AIIS', 'AG', 'AF']

def post_processing(df_predict):
    df_predict = transform_df(df_predict)
    list_df_huruf_sebelum_angka = list(df_predict['huruf_sebelum_angka'])
    list_df_huruf_setelah_angka = list(df_predict['huruf_setelah_angka'])
    list_df_semua_angka = list(df_predict['semua_angka'])
    list_df_len_huruf_sebelum_angka = list(df_predict['len_huruf_sebelum_angka'])
    list_df_len_huruf_setelah_angka = list(df_predict['len_huruf_setelah_angka'])
    list_df_len_semua_angka = list(df_predict['len_semua_angka'])
    for i in range(len(df_predict)):
        if (list_df_huruf_sebelum_angka[i] not in list_huruf_sebelum_angka) or (list_df_huruf_sebelum_angka[i] is None):
            list_df_huruf_sebelum_angka[i] = 'B'
        if list_df_len_huruf_setelah_angka[i]>3:
            list_df_huruf_setelah_angka[i] = list_df_huruf_setelah_angka[i][:3]
        if (list_df_huruf_sebelum_angka[i] in ['A','AB','AA','AE','AF']) and (list_df_len_huruf_setelah_angka[i]>2):
            list_df_huruf_setelah_angka[i] = list_df_huruf_setelah_angka[i][:2]
        elif list_df_len_huruf_setelah_angka[i]==0:
            list_df_huruf_setelah_angka[i] = ''
        if list_df_len_semua_angka[i]>4:
            list_df_semua_angka[i] = list_df_semua_angka[i][:4]
        elif list_df_len_semua_angka[i]==0:
            list_df_semua_angka[i] = ''
    df_predict['mod_huruf_sebelum_angka'] = list_df_huruf_sebelum_angka
    df_predict['mod_huruf_setelah_angka'] = list_df_huruf_setelah_angka
    df_predict['mod_semua_angka'] = list_df_semua_angka
    df_predict['mod_labels'] = df_predict.apply(lambda row: ''.join([row['mod_huruf_sebelum_angka'], row['mod_semua_angka'], row['mod_huruf_setelah_angka']]), axis=1)
    return df_predict