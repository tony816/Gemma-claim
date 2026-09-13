"""Temporary, bounded Windows idle-sleep prevention; display may turn off."""
import ctypes,json,os,time
from .core import OUT,write


def main():
    assert os.name=='nt'
    record=json.loads((OUT/'pod_created.json').read_text(encoding='utf-8'))
    deadline=record['deadline']+1800
    fn=ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes=[ctypes.c_uint];fn.restype=ctypes.c_uint
    assert fn(0x80000001),'temporary_sleep_prevention_failed'
    write('sleep_prevention.json',{'pid':os.getpid(),'started_at':time.time(),'deadline':deadline,
        'temporary':True,'display_kept_on':False,'permanent_power_settings_changed':False})
    try:
        while time.time()<deadline and not (OUT/'final_result.json').exists():time.sleep(10)
    finally:
        fn(0x80000000)
        write('sleep_prevention_released.json',{'time':time.time(),'pid':os.getpid()})

if __name__=='__main__':main()
