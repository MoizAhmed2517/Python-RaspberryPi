# Python-RaspberryPi-RS_485_Communication

A script capable of extracting any data from any Energy Analyzer.
_*Note:* the above can be used for extracting data via RS-485 communication. We are using AccuEnergy L-model for mapping RS-485 table._

To use this code in any other energy meter or any sensor for RS-485 communication you need to change the following:

1. Update the params according to your analyzer or sensor.
2. Update the gainParam dictionary. (_Note: Energy analyzer or sensor are map on registers in memory unit. To make memory constant size of the value to be stored needs to be same. Since, every variable size varies. Therefore, additional gain is added to make register size same in the memory_.)
3. Update the header. It will automatically update the CSV header.

Note: This program is extendable to any storage mechanism whether databases, .txt files, or .json format.
