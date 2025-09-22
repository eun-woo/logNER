import re
import regex
from collections import defaultdict
original_sentence = ['DIR*', 'completeFile', ':', '/155.230.91.227', ':', '5000', 'by', 'DFSClient_NONMAPREDUCE_958626130_1']

log_span = defaultdict(str)
for i in range(len(original_sentence)):
    for j in range(i, len(original_sentence)):
        if i < j:
            log_span[(i, j)] = log_span[(i, j-1)] + " " + original_sentence[j]
        else:
            log_span[(i, j)] = log_span[(i, j)] + original_sentence[j]
# hostport 정규표현식 정의 
#Matched: 192.168.1.1:8080
#Matched: example.com:443
HOSTPORT = f"(?:(?:(?:[0-9A-Za-z][0-9A-Za-z-]{0,62})(?:\.(?:[0-9A-Za-z][0-9A-Za-z-]{0,62}))*(\.?|\b))|(?:(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})\.){3}(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})(?![0-9]))):\b(?:[1-9][0-9]*)\b"

# 정규표현식 정의
USERNAME = r"[a-zA-Z0-9._-]+"
USER = r"{USERNAME}".format(USERNAME=USERNAME)
UUID = r"[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}"
HOSTNAME = r"\b(?:[0-9A-Za-z][0-9A-Za-z-]{0,62})(?:\.(?:[0-9A-Za-z][0-9A-Za-z-]{0,62}))*(\.?|\b)"
IPV4 = r"(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2}))(?![0-9])" 
IPV6 = r"((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?"
IP = r"(?:{IPV6}|{IPV4})".format(IPV4 = IPV4, IPV6 = IPV6)
POSINT = r"\b(?:[1-9][0-9]*)\b"
IPPORT = r"{IP}:{PORT}".format(IP=IP, PORT=POSINT)
IPORHOST = r"(?:{HOSTNAME}|{IP})".format(HOSTNAME=HOSTNAME, IP=IP)

# paths
UNIXPATH = r"(?>/(?>[\w_%!$@.-]+|\\.)*)+"
WINPATH = r"(?>[A-Za-z]+:|\\)(?:\\[^\\?*]*)+"
BGLPATH = r"^(/[a-zA-Z0-9\-_\.]+)+(/[a-zA-Z0-9\-_\.]+)*$"
PATH = r"(?:{UNIXPATH}|{WINPATH}|{BGLPATH})".format(UNIXPATH=UNIXPATH, WINPATH=WINPATH, BGLPATH=BGLPATH)
PATHPORT = r"{PATH}:{PORT}".format(PATH=PATH, PORT=POSINT)
URIPATH = r"(?:/[A-Za-z0-9$.+!*'(){},~:;=@#%_\-]*)+"
URIPARAM = r"\?[A-Za-z0-9$.+!*'|(){},~@#%&/=:;_?\-\[\]]*"
URIPATHPARAM =r"{URIPATH}(?:{URIPARAM})?".format(URIPATH=URIPATH, URIPARAM=URIPARAM)
URIHOST = r"{IPORHOST}(?::{POSINT})?".format(IPORHOST=IPORHOST, POSINT=POSINT)
URIPROTO = r"[A-Za-z]+(\+[A-Za-z+]+)?"
URI = r"{URIPROTO}://(?:{USER}(?::[^@]*)?@)?(?:{URIHOST})?(?:{URIPATHPARAM})?".format(URIPROTO=URIPROTO, USER=USER, URIHOST=URIHOST, URIPATHPARAM=URIPATHPARAM)
# # 정규표현식을 이용한 추출
# for span in log_span.keys():
#     match = re.fullmatch(IP, log_span[span])
#     if match:
#         print(f"Matched: {log_span[span]}")
#         if span[1]+1

def find_regex(tokenized_log):
    regex_rng=[] 
    for i in range(len(original_sentence)):
        for j in range(i, len(original_sentence)):
            match_URI = re.fullmatch(URI, log_span[(i, j)])
            if match_URI:
                print(f"URI: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
        
            match_IP = re.fullmatch(IP, log_span[(i, j)])
            if match_IP:
                print(f"IP: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                if original_sentence[j+1] == ":" and re.fullmatch(POSINT, original_sentence[j+2]):
                    print(f"PORT: {original_sentence[j+2]}")
                    regex_rng.append((j+2, j+2, 0.999))
                    print(f"IPPORT: {log_span[(i, j+2)]}")
                    regex_rng.append((i, j+2, 0.999))
                continue
            
            # match_IPPORT = re.fullmatch(IPPORT, log_span[(i, j)])
            # if match_IPPORT:
            #     print(f"IP: {log_span[(i, j-2)]}  PORT: {log_span[(j, j)]}")
            #     # regex_rng.append((i, j-2, 0.999))
            #     regex_rng.append((j, j, 0.999))
            #     continue

            match_PATH = regex.fullmatch(PATH, log_span[(i, j)])
            if match_PATH:
                print(f"PATH: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                if original_sentence[j+1] == ":" and re.fullmatch(POSINT, original_sentence[j+2]):
                    print(f"PORT: {original_sentence[j+2]}")
                    regex_rng.append((j+2, j+2, 0.999))
                    print(f"PATHPORT: {log_span[(i, j+2)]}")
                    regex_rng.append((i, j+2, 0.999))
                continue
            
            # match_PATHPORT = regex.fullmatch(PATHPORT, log_span[(i, j)])
            # if match_PATHPORT:
            #     print(f"PATH: {log_span[(i, j-2)]}  PORT: {log_span[(j, j)]}")
            #     # regex_rng.append((i, j-2, 0.999))
            #     regex_rng.append((j, j, 0.999))
            #     continue

            match_UUID = re.fullmatch(UUID, log_span[(i, j)])
            if match_UUID:
                print(f"UUID: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
    return regex_rng

print(log_span)
print(find_regex(original_sentence))
ex = "/user/root/rand/_temporary/_task_200811092030_0001_m_001648_0/part-01648"
m = regex.fullmatch(PATH, ex)
if m:
    print(f"matched: ", ex)
    
