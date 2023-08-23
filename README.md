# furusato_recommend

def five_core(df):
    df=df[df['remap_id'].isin(df['remap_id'].value_counts()[df['remap_id'].value_counts()>=5].index)]
    df=df[df['customer_id'].isin(df['customer_id'].value_counts()[df['customer_id'].value_counts()>=5].index)]
    return df

def ten_core(df):
    df=df[df['remap_id'].isin(df['remap_id'].value_counts()[df['remap_id'].value_counts()>=10].index)]
    df=df[df['customer_id'].isin(df['customer_id'].value_counts()[df['customer_id'].value_counts()>=10].index)]
    return df

python conversion_tools/run.py --input_path ../recbole/ --ouypuy_path ../recbole --convert_inter --convert_user --convert_item
