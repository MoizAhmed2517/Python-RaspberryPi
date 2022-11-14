from pymodbus.client.sync import ModbusSerialClient as ModbusClient
from datetime import datetime
import csv
import time

client = ModbusClient(method='rtu', port='/dev/ttyUSB0', baudrate=19200, timeout=3)

client.connect()

params = {'F': 304, 'U1': 305, 'U2': 306, 'U3': 307, 'U12': 308, 'U23': 309, 'U31': 310,
          'I1': 311, 'I2': 312, 'I3': 313, 'In': 314, 'Pa': 315, 'Pb': 316, 'Pc': 317,
          'Psum': 318, 'Qa': 319, 'Qb': 320, 'Qc': 321, 'Qsum': 322, 'PFa': 324, 'PFb': 325,
          'PFc': 326, 'PFsum': 327, 'U-unbl': 328, 'I_unbl': 329, 'L/C/R': 330}

gainParam = {'F': 100, 'U1': 10, 'U2': 10, 'U3': 10, 'U12': 10, 'U23': 10, 'U31': 10,
             'I1': 50, 'I2': 50, 'I3': 50, 'In': 50, 'Pa': 50, 'Pb': 50, 'Pc': 50,
             'Psum': 50, 'Qa': 50, 'Qb': 50, 'Qc': 50, 'Qsum': 50, 'PFa': 1000, 'PFb': 1000,
             'PFc': 1000, 'PFsum': 1000, 'U-unbl': 1000, 'I_unbl': 1000, 'L/C/R': 1}

header = ['Date Time', 'Frequency', 'Voltage_L1', 'Voltage_L2', 'Voltage_L3', 'Voltage_12', 'Voltage_23', 'Voltage_31',
          'Current_I1',
          'Current_I2', 'Current_I3', 'Current_In', 'Power_Pa', 'Power_Pb', 'Power_Pc', 'PSum', 'Reactive_Pa',
          'Reactive_Pb', 'Reactive_Pc', 'Qsum', 'Pf_a', 'Pf_b', 'Pf_c', 'Pfsum', 'U_unbl(%)', 'I_unbl(%)',
          'L(76)C(67)R(82)']

para = []

with open('electrical_parameters.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header)
    while True:
        now = datetime.now()
        df_string = now.strftime("%d/%m/%Y %I:%M:%S")
        para.append(df_string)
        for key, gain in zip(params.keys(), gainParam.values()):
            read = client.read_holding_registers(address=params[key], count=1, unit=0)
            try:
                if len(para) != 27:
                    val = read.registers[0] / gain
                    para.append(val)
            except:
                print(read)
        writer.writerow(para)
        para.clear()
        time.sleep(0.35)