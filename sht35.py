import smbus
import time

# SHT35的I2C地址
SHT35_ADDR = 0x44

# SHT35的命令
SHT35_CMD_READ_HIGH = 0x2C06
SHT35_CMD_READ_MEDIUM = 0x2C0D
SHT35_CMD_READ_LOW = 0x2C10

# 创建I2C实例
i2c = smbus.SMBus(1)

def read_sht35_data(cmd=SHT35_CMD_READ_HIGH):
    # 发送测量命令
    i2c.write_i2c_block_data(SHT35_ADDR, cmd >> 8, [cmd & 0xFF])
    
    # 等待测量完成
    time.sleep(0.1)
    
    # 读取6个字节的数据
    data = i2c.read_i2c_block_data(SHT35_ADDR, 0x00, 6)
    
    # 转换温湿度数据
    temp = (((data[0] << 8) | data[1]) * 175 / 65535.0) - 45
    humi = (((data[3] << 8) | data[4]) * 100 / 65535.0)
    
    return temp, humi

# 主程序
while True:
    temperature, humidity = read_sht35_data()
    print(f"Temperature: {temperature:.2f}°C, Humidity: {humidity:.2f}%")
    time.sleep(1)