import requests
from scapy.all import getmacbyip
def get_mac(ip):
    target_mac = getmacbyip(ip)
    while not target_mac:
        target_mac = getmacbyip(ip)
    print(target_mac)
    return target_mac

def post_data(url,data, name):
    try:
        x = requests.post(url, json=data)
        print(name, x)
        if x.status_code == 200:
            return True
        return False
    except:
        print("Some problems occursd during sending the data")
