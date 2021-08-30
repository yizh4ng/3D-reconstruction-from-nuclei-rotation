import os
from roma import finder
from roma import console


src_path = os.path.abspath('.')
dst_path = r'\\172.16.233.191\wmshare\projects\lambai'
# dst_path = r'\\172.16.233.191\xinshare\projects\lambai'
# dst_path = r'\\LAPTOP-U4LIQLV8\Users\lambc\Dropbox\project\lambai'

ignored_patterns=('.*', '__*__', 'checkpoints', 'logs')
finder.synchronize(src_path, dst_path, pattern='*.py',
                   ignored_patterns=ignored_patterns, verbose=True)