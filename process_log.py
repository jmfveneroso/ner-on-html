import sys
import re

def print_model(attrs):
  if len(attrs) == 0:
    return

  f  = '+F' if attrs['V-Use features'] == 'True' else ''
  st = '+ST' if attrs['V-Self train'] == 'True' else ''

  s = attrs['Model'] + f + st + '\t'
  s += attrs['V-Precision'] + '\t'
  s += attrs['V-Recall'] + '\t'
  s += attrs['V-F1'] + '\t'
  s += attrs['V-Accuracy'] + '\t'
  s += attrs['T-Precision'] + '\t'
  s += attrs['T-Recall'] + '\t'
  s += attrs['T-F1'] + '\t'
  s += attrs['T-Accuracy'] + '\t'
  s += attrs['Time']
  print(s)

with open(sys.argv[1], 'r') as f:
  is_validate = True
  attrs = {}
  lines = [l.strip() for l in f if not re.match(r'^[A-Z]', l) is None]
  for l in lines:
    if l.startswith('Model'):
      print_model(attrs)
      attrs = {'Model': l.split(': ')[1]}
      is_validate = True
    elif l.startswith('Validate'):
      pass
    elif l.startswith('Test'):
      is_validate = False
    elif l.startswith('Time elapsed'):
      x = l.split(' ')
      attrs['Time'] = x[2]
    else:
      x = l.split(': ')
      mark = 'V-' if is_validate else 'T-'
      attrs[mark + x[0]] = x[1]
  print_model(attrs)
