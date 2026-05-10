import csv

labels = {
    '13': 'positive',
    '14': 'positive',
    '15': 'positive',
    '16': 'positive',
    '17': 'neutral',
    '18': 'neutral',
    '19': 'positive',
    '21': 'positive',
    '23': 'positive',
    '25': 'neutral',
    '26': 'positive',
    '27': 'neutral',
    '28': 'neutral',
    '29': 'neutral',
    '30': 'neutral',
    '31': 'neutral',
    '32': 'neutral',
    '33': 'neutral',
    '34': 'neutral',
    '35': 'positive',
    '36': 'neutral',
    '37': 'neutral',
    '38': 'neutral',
    '39': 'neutral',
    '40': 'neutral',
    '41': 'neutral',
    '42': 'neutral',
    '43': 'neutral',
    '44': 'neutral',
    '45': 'neutral',
    '46': 'positive',
    '47': 'positive',
    '48': 'neutral',
    '49': 'neutral',
    '50': 'neutral',
    '51': 'neutral',
    '52': 'neutral',
    '53': 'positive',
    '54': 'neutral',
    '55': 'neutral',
    '56': 'neutral',
    '57': 'neutral',
    '58': 'neutral',
    '59': 'neutral',
    '60': 'neutral',
    '61': 'neutral',
    '62': 'negative',
    '63': 'neutral',
    '64': 'neutral',
    '65': 'positive',
    '66': 'positive',
    '67': 'positive',
    '68': 'neutral',
    '69': 'positive',
    '70': 'positive',
    '71': 'neutral',
    '72': 'neutral',
    '73': 'neutral',
    '74': 'negative',
    '75': 'neutral'
}

path = 'data/manual_labeling_finbert.csv'
rows = list(csv.DictReader(open(path, encoding='utf-8')))

for row in rows:
    if row['sentiment_result_id'] in labels and not row['manual_label']:
        row['manual_label'] = labels[row['sentiment_result_id']]

with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
