import re, json
log = open('logs/logs', encoding='utf-8').read()
blocks = re.findall(r"```json\s*(.*?)```", log, re.DOTALL)
for b in reversed(blocks):
    try: d = json.loads(b)
    except: continue
    if any('Wall' in k or 'Building' in k for k in d) and \
       any(isinstance(v, dict) and v.get('groups') for v in d.values()):
        print("TASKS:", list(d.keys()))
        json.dump(d, open('_replay_cls.json', 'w'))
        print("saved")
        break
