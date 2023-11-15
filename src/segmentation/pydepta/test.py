from pydepta import Depta
d = Depta()
url = 'https://search.yahoo.com/search;_ylt=AwrEwhFGyDNik_IAP2pXNyoA;_ylu=Y29sbwNiZjEEcG9zAzEEdnRpZANBMDYzNF8xBHNlYwNwYWdpbmF0aW9u?p=einstein&pz=7&fr=sfp&fr2=p%3As%2Cv%3Asfp%2Cm%3Asb-top&bct=0&b=50&pz=7&bct=0&xargs=0'
regions = d.extract(url=url)
for region in regions:
    print('------------------------------------------------------------------------------------------')
    for record in region.as_plain_texts():
        print(record)