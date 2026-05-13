#-*- coding: utf-8 -*-
import os
import struct
import time
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_RCVBUF

class KLib():
    def __init__(self,_server_ip = "127.0.0.1", _port = 3800):
        self.server_ip = _server_ip
        self.port = _port
        self.nrow = 0
        self.ncol = 0
        # 수신받을 데이터가 length보다 크다면, 해당 length를 수정하여야 한다.
        self.totalPacketSize = 40000
        self.datasize = 40000        
        self.dataType = "Raw"
        self.dataMatrix = []
        self.buf = None
        self.client_socket = None
        self.client_socket_connection = False
        self.staticPacketLen = 96

    def init_connection(self):
        try:
            self.addr = (self.server_ip, self.port)
            self.client_socket = socket(AF_INET, SOCK_STREAM)
            self.client_socket.setsockopt(SOL_SOCKET, SO_RCVBUF, 50000)
            self.client_socket.connect(self.addr)
        except Exception as e:
            print('Failed to connect TCP/IP!')
            self.client_socket_connection = False
            return False
        self.client_socket_connection = True
        return True

    #TcpIP 연결 시도
    def init(self):
        if self.init_connection() is False:
            return

        resp = self.client_socket.recv(self.totalPacketSize) #버퍼 받기

        self.buf = resp
        
        sp = 0
        

        #header가 2개 이상이 아닌경우 패킷이 다안들어왔을 가능성이 있음
        while(1):
            if(len(self.buf) >= self.totalPacketSize):
                break
            resp = self.client_socket.recv(self.totalPacketSize)
            self.buf = self.buf + resp

             #header 위치 찾기            
            while(1):
                sp = self.buf.index(0x7e,sp)
                if(self.buf[sp+1] == 0x7e and self.buf[sp+2]== 0x7e and self.buf[sp+3] == 0x7e):      
                    # Header,Tail의 크기를 뺀 PackestSize가 들어온다                                  
                    self.totalPacketSize = int.from_bytes(self.buf[4:7],byteorder='little') + 8
                    self.nrow = int.from_bytes(self.buf[88:91],byteorder='little')
                    self.ncol = int.from_bytes(self.buf[92:95],byteorder='little')
                    self.datasize = self.nrow * self.ncol
                    break

        self.device = self.buf[4:28]
        self.sensor = self.buf[28:52]
        self.nrow = int.from_bytes(self.buf[88:91],byteorder='little')
        self.ncol = int.from_bytes(self.buf[92:95],byteorder='little')
        self.bufSize = self.nrow * self.ncol


        # 초기 통신 시 length 크기가 datasize + 200 보다 작다면 Raw ADC 데이터로 판별
        # 아니라면 Force 데이터로 판별함 (Force의 경우 8byte 통신으로 데이터 크기가 크다)
        if(self.datasize + 200 >= self.totalPacketSize):
            self.dataType = "Raw"
        else:
            self.dataType = "Force"        
            self.bufSize *=8
       
        
         #header, tail을 뺀 버퍼를 result에 집어넣음
        self.result = self.buf[sp + 4 : sp + self.totalPacketSize]

        # dataMatrix array 생성
        if self.dataType == "Raw":
            for i in range(96,self.bufSize+96):
                self.dataMatrix.append(int(self.buf[i]))
        else:
            for i in range(96,self.bufSize+96,8):
                self.dataMatrix.append(struct.unpack('d',self.buf[i:i+8])[0])
        
        #Fix this code
        self.buf = b''

    def check_tcp_connection(self):
        if(self.client_socket_connection == True):
            return True
        else:
            return False
        
    #서버와 tcp 연결 시도
    def start(self):
        self.init()

    #서봐의 tcp 연결 끊기
    def stop(self):
        self.client_socket.close()
        self.client_socket_connection = False

    #패킷읽기
    def read(self):
        self.buf  = self.buf + self.client_socket.recv(self.totalPacketSize)

        #header 검색
        while(1):
            #Fix this code
            if(len(self.buf) > self.totalPacketSize):
                break
            #Fix this code
            resp = self.client_socket.recv(self.totalPacketSize)
            self.buf = self.buf + resp
        
        #header 위치 찾기
        sp = 0
        while(1):
            try:
                sp = self.buf.index(0x7e,sp)
            except error:
                self.buf = None
                print("Error buf None")
                return
            
            if(sp+self.totalPacketSize>len(self.buf)):
                self.buf = self.buf[sp:]
                return
            
            if(self.buf[sp+1] == 0x7e and self.buf[sp+2]== 0x7e and self.buf[sp+3] == 0x7e):
                break
        
       

          # dataMatrix array 생성
        if self.dataType == "Raw":
            for i in range(self.staticPacketLen+sp,self.bufSize+self.staticPacketLen+sp):
                self.dataMatrix[i-self.staticPacketLen-sp] = int(self.buf[i])
        else:
            for i in range(self.staticPacketLen+sp, self.bufSize+self.staticPacketLen+sp, 8):
                index = int((i-self.staticPacketLen-sp) / 8)
                self.dataMatrix[index] = (struct.unpack('d', self.buf[i:i+8])[0])
        
        
        # 읽어들인 adc 데이터 부분 삭제
        self.buf = self.buf[sp+self.totalPacketSize:]

        # 수신 버퍼에 데이터가 10frame 이상 쌓일경우 clear
        if len(self.buf) > self.totalPacketSize * 3:
            self.buf = b''

    def printData(self):
        os.system('cls')
        if self.dataType == "Raw":
            for i in range(self.nrow):
                write_str = ""
                for j in range(self.ncol):
                    write_str = write_str + " " + str(self.dataMatrix[i*self.ncol + j])
                print(write_str)
        else:
            for i in range(self.nrow):
                write_str = ""
                for j in range(self.ncol):
                    #소수점 3자리까지 표현
                    value = self.dataMatrix[i*self.ncol + j]
                    write_str += f" {value:.3f}"
                print(write_str)
        print()

if __name__ == "__main__":
    klib = KLib("127.0.0.1", 3800)
    tick = 0
    FPS = 0
    prevTime = time.time()
    
    klib.start()
    while(1):
        klib.read()    
        klib.printData()

        tick = tick + 1
        #FPS 계산
        curTime = time.time()
        if curTime - prevTime > 1 :            
            FPS = tick
            prevTime = curTime
            tick = 0
        print("FPS : ", FPS)
        
