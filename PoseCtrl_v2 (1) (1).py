import socket
import csv
from datetime import datetime



class PoseCtrl(object):
    
    def __init__(self, ip, port,lock):
        super().__init__()
        self.IP = ip
        self.PORT = int(port)
        self.lock = lock

        self.Connected = False
        self.Homed = False
        self.socketClient = None

        self.sendedData=""
        self.recievedData=""
        self.waitingData = False
        
        self.MCError = False
        self.PLCError = False
        self.ECError = False
        self.EmergencyStop = False
        self.DistanceProtection = False
        self.MS2_Limit = False
        self.Error = False
        self.MCE_code = ""
        self.PLCE_code = ""
        self.ECE_code = ""

        self.Velocity = 0
        self.Acceleration = 0
        self.Deceleration = 0

        self.MS1Pose = []
        self.MS2Pose = []
        
        self.MS1CPose = 0
        self.MS2CPose = 0
        self.Fire = False
        
        self.ScanMode = False

        self.CrcCount = 0
        self.CrcMaxCount = 5    
        
        self.errorLogFile_address = ""

        self.temp = 30
        #print("Created PLC Object")
        
        
    def __del__(self):
        pass
       
        
           
    def connect(self):
        try:
            self.socketClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socketClient.settimeout(15)
            self.socketClient.connect((self.IP,self.PORT))            
            self.Connected = True           
            return [self.Connected, "Success"]
        except socket.error as er:
            self.Connected = False
            print("PoseCtrl Communication Error [Connection]: %s" % er)            
            return [self.Connected, str(er)]
        
    def check_connection(self):
        """
        Check if the PLC is still connected without attempting to reconnect.
        Returns True if connected, False otherwise.
        """
        if self.Connected:
            try:
                # Send a byte to verify the connection
                self.socketClient.send(b'\x00')  # Sending a single null byte
                return [True, "PLC Connected"]
            except socket.error:
                # If an error occurs, mark as disconnected
                self.Connected = False
                self.socketClient = None
                return [False, "Connection lost"]
        return [False, "Connection lost"]  # Not connected in the first place
    
    def disconnect(self):        
        if self.socketClient is not None:
            try:
                self.socketClient.close()
                self.Connected = False
                print("PoseCtrl Connection is closed.")
            except socket.error as er:
                print("PoseCtrl Communication Error [Disconnection]: %s" % er)
                self.Connected = False
        else:
             print("PoseCtrl Communication Error [Disconnection]: There is no Connection")
             self.Connected = False
    
                
    def getVel(self):
        VelMSG = [8,0,0,0,0] #0x08
        VelMSG_B = bytearray(VelMSG)
        crc = self.crc16(VelMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        VelMSG_B.append(crcB[1])
        VelMSG_B.append(crcB[0])
        self.sendedData = VelMSG_B
        print("Get Vel :" + str(self.sendedData))
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()
    
    def ReadEncoder(self):
        EncMSG = [2,0,0,0,0] #0x08
        EncMSG_B = bytearray(EncMSG)
        crc = self.crc16(EncMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        EncMSG_B.append(crcB[1])
        EncMSG_B.append(crcB[0])
        self.sendedData = EncMSG_B
        print("Enc Vel:" + str(self.sendedData))
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()
    
        
    
    
    def getAc(self):
        AcMSG = [9,0,0,0,0] #0x08
        AcMSG_B = bytearray(AcMSG)
        crc = self.crc16(AcMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        AcMSG_B.append(crcB[1])
        AcMSG_B.append(crcB[0])
        self.sendedData = AcMSG_B
        print("Get Ac :" + str(self.sendedData))
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()
    
    def getDec(self):
        DecMSG = [10,0,0,0,0] #0x08
        DecMSG_B = bytearray(DecMSG)
        crc = self.crc16(DecMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        DecMSG_B.append(crcB[1])
        DecMSG_B.append(crcB[0])
        self.sendedData = DecMSG_B
        print("Get Dec :" + str(self.sendedData))
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()
        
    def setVel(self,Vel:int):
        vel_b = Vel.to_bytes(2, "big")
        VelMSG = [5,0,0,vel_b[0],vel_b[1]] #0x08
        VelMSG_B = bytearray(VelMSG)
        crc = self.crc16(VelMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        VelMSG_B.append(crcB[1])
        VelMSG_B.append(crcB[0])
        self.sendedData = VelMSG_B
        print("SetVelSend:" + str(VelMSG_B))
        try:
            self.socketClient.send(self.sendedData) 
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()
    def setAc(self,Ac:int):
        Ac_b = Ac.to_bytes(2, "big")
        AcMSG = [6,0,0,Ac_b[0],Ac_b[1]] #0x08
        AcMSG_B = bytearray(AcMSG)
        crc = self.crc16(AcMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        AcMSG_B.append(crcB[1])
        AcMSG_B.append(crcB[0])
        self.sendedData = AcMSG_B
        print("SetAcSend:" + str(AcMSG_B))
        try:
            self.socketClient.send(self.sendedData) 
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()    
    
    def setDec(self,Dec:int):
        dec_b = Dec.to_bytes(2, "big")
        DecMSG = [7,0,0,dec_b[0],dec_b[1]] #0x08
        DecMSG_B = bytearray(DecMSG)
        crc = self.crc16(DecMSG_B,5)
        crcB = crc.to_bytes(2, 'big') 
        DecMSG_B.append(crcB[1])
        DecMSG_B.append(crcB[0])
        self.sendedData = DecMSG_B
        print("SetDecSend:" + str(DecMSG_B))
        try:
            self.socketClient.send(self.sendedData) 
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"
        return self.recieve()        
        
    def send(self,data):        
        crc = self.crc16(data,5)
        crcB = crc.to_bytes(2, 'big')   
        #data_ =[data[0], data[1], data[2], data[3], data[4], crcB[1],crcB[0]]
        data.append(crcB[1])
        data.append(crcB[0])
        print(data)
        self.sendedData = data
        try:
            self.socketClient.send(data)
            return True
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Error[Send MSG]: %s" % er)
            return False

    def sendPose(self,MS1Pose: int,MS2Pose: int):
        MS1_poseB = (MS1Pose).to_bytes(2, 'big')
        MS2_poseB = (MS2Pose).to_bytes(2, 'big')
        PoseMSG= [1,MS1_poseB[0],MS1_poseB[1], MS2_poseB[0], MS2_poseB[1]]
        PoseMSG_B = bytearray(PoseMSG)        
        crc = self.crc16(PoseMSG_B,5)
        crcB = crc.to_bytes(2, 'big')
        PoseMSG_B.append(crcB[1])
        PoseMSG_B.append(crcB[0])        
        self.sendedData = PoseMSG_B
        print("Sending Pose:" + str(self.sendedData))
        try:
            self.socketClient.send(self.sendedData)
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"

        return self.recieve()

        
        
    def sendHome(self):        
        HomeMSG = [170,0,0,0,0] #0xAA
        HomeMSG_B = bytearray(HomeMSG)        
        crc = self.crc16(HomeMSG_B,5)
        crcB = crc.to_bytes(2, 'big')
        HomeMSG_B.append(crcB[1])
        HomeMSG_B.append(crcB[0])
        print("SendHome:" + str(HomeMSG_B))
        self.sendedData = HomeMSG_B
        self.Homed = "Waiting" 
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.Homed = False
            self.log(str(er))
            print("PoseCtrl Communication Error[Pose Sending]: %s" % er)
            return "Error","Communication"

        return self.recieve()

        
                    
        
    def GetError(self,askedErr):        
        ErrMSG= [238,0,0,0,0] #0xEE
        ErrMSG_B = bytearray(ErrMSG)        
        crc = self.crc16(ErrMSG_B,5)
        crcB = crc.to_bytes(2, 'big')
        ErrMSG_B.append(crcB[1])
        ErrMSG_B.append(crcB[0])
        print("SendGE:" + str(ErrMSG_B))
        self.sendedData = ErrMSG_B
        try:   
            
            self.socketClient.send(self.sendedData)            
        except socket.error as er:
            self.log(str(er))            
            print("PoseCtrl Communication Error [GetError]: %s" % er) 
            return "Error","Communication"
        return self.recieve()
            
        
        
    def GetErrorCode(self,AskedErr: int):        
        ErrMSG= [14,AskedErr,0,0,0] #0x0E
        ErrMSG_B = bytearray(ErrMSG)        
        crc = self.crc16(ErrMSG_B,5)
        crcB = crc.to_bytes(2, 'big')
        ErrMSG_B.append(crcB[1])
        ErrMSG_B.append(crcB[0])
        print("GECSend:" + str(ErrMSG_B))
        self.sendedData = ErrMSG_B
        #self.socketClient.send(self.sendedData)
        try:    
            self.socketClient.send(self.sendedData)            
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error [GetErrorCode]: %s" % er)
            return "Error","Communication"
        return self.recieve()
      
        
    def ResetErrors(self):
        ResetMSG = [3,0,0,0,0] #0x03
        ResetMSG_B = bytearray(ResetMSG)
        crc = self.crc16(ResetMSG_B, 5)
        crcB = crc.to_bytes(2, 'big')
        ResetMSG_B.append(crcB[1])
        ResetMSG_B.append(crcB[0])
        print("RESend:" + str(ResetMSG_B))
        self.sendedData = ResetMSG_B

        try:
            self.socketClient.send(self.sendedData)
        except socket.error as er:
            self.log(str(er))
            print("PoseCtrl Communication Error [ResetErrorCode]: %s" % er)
            return "Error","Communication"
        response = self.recieve()
        if (response[1] == "NoError"):
            return "Error","Reseted"
        return response
        
    def SetF(self):
               
        FMSG = [11,0,0,0,0] #0x0B
        FMSG_B = bytearray(FMSG)        
        crc = self.crc16(FMSG_B,5)
        crcB = crc.to_bytes(2, 'big')
        FMSG_B.append(crcB[1])
        FMSG_B.append(crcB[0])
        #print("SendHome:" + str(TempMSG_B))
        self.sendedData = FMSG_B
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.logger.error(str(er))
            #print("PoseCtrl Communication Error[Get Temp]: %s" % er)
            return "Error","Communication"
    
        return self.recieve()
    
    def ResetF(self):
             
        FMSG = [12,0,0,0,0] #0x0C
        FMSG_B = bytearray(FMSG)        
        crc = self.crc16(FMSG_B,5)
        crcB = crc.to_bytes(2, 'big')
        FMSG_B.append(crcB[1])
        FMSG_B.append(crcB[0])
        
        self.sendedData = FMSG_B
        try:
            self.socketClient.send(self.sendedData)           
        except socket.error as er:
            self.logger.error(str(er))
            #print("PoseCtrl Communication Error[Get Temp]: %s" % er)
            return "Error","Communication"
    
        return self.recieve()

    def crc16(self,data : bytearray, length):
        crc = 0xFFFF
        for i in range(0, length):
            crc ^= data[i]
            for j in range(8,0,-1):
                if (crc & 0x0001) != 0:
                    crc =(crc >> 1) ^ 0xA001
                else:
                        crc = crc >> 1
        return crc 
    
    def recieve(self):
        try:
            data = self.socketClient.recv(7)
        except socket.error as error:
            self.log(str(error))
            print ("PoseCtrl Communication Error [Recieve] %s" % error)
            self.waitingData = False
            return "Error","Communication"
        self.recievedData = data
        print("Receive:" + self.recievedData.hex() + " Data len:"+str(len(self.recievedData)))
        if len(self.recievedData)<=0:
            self.Connected = False
            self.waitingData = False
            return "Error","PCtrlDisconnection"
        self.waitingData = False
        return self.makeSenseMSG(data)
        
    def askErrCode(self,msg):
        #print("askErr" + str(msg))
        Err_bit = f'{msg[1]:08b}'    #0000 1= (MS2 Limit) 1 =(ECE)  1 = (PLC) 1= (MCE)
        # print(str(Err_bit))
        self.MCE_code = "-"
        self.PLCE_code = "-"
        self.ECE_code = "-"
        for i in [1,2,3,4]:   
            # print(str(Err_bit[8-i]))
            if int(Err_bit[8-i]) == 1:
                code = self.GetErrorCode(i)
                self.log(str(code))
                print("Code:" + str(code))
                #self.recieve()
                

    def makeSenseMSG(self,data):
        crc = self.crc16(data,5)
        crcB = crc.to_bytes(2, 'big')
        if (crcB[1] != data[5]) or (crcB[0] != data[6]):
            self.CrcCount += 1##### muhtemel problem var çözemedim :()
            if self.CrcCount < self.CrcMaxCount:
                print("CRC Error")
                self.socketClient.send(self.sendedData)
                self.recieve()
                #return "Crc Error trying again"
            else:                
                self.CrcCount = 0
                self.log("CRC")
                return "Error","CRC" 
                   
        if data[0]==int.from_bytes(b'\x01', "big"):          
            Ms1p = 256*data[1] + data[2]
            Ms2p = 256*data[3] + data[4]
           
            return "Start Pose:"+str(Ms1p) + ":" + str(Ms2p)
            
        elif data[0] == int.from_bytes(b'\x02', "big"): 
            Ms1p = 256*data[1] + data[2]
            Ms2p = 256*data[3] + data[4]
            self.MS1CPose = str(Ms1p)
            self.MS2CPose = str(Ms2p)
            
            return "Current Pose :"+str(Ms1p) + ":" + str(Ms2p)
        
        elif data[0] == int.from_bytes(b'\x03', "big"):
            pass
        
        elif data[0] == int.from_bytes(b'\x04', "big"): 
            pass  
        
        elif data[0] == int.from_bytes(b'\x05', "big"):
            self.Velocity = 256*data[3] + data[4]
            msg_return = str(self.Velocity)
            return  "Vel",msg_return
                    
        elif data[0] == int.from_bytes(b'\x06', "big"): 
            self.Acceleration = 256*data[3] + data[4]
            msg_return = str(self.Acceleration)
            return "Ac",msg_return
        
        elif data[0] == int.from_bytes(b'\x07', "big"): 
            self.Deceleration = 256*data[3] + data[4]
            msg_return = str(self.Deceleration) 
            return "Dec", msg_return
        
        elif data[0] == int.from_bytes(b'\x08', "big"): 
            pass 
        
        elif data[0] == int.from_bytes(b'\x09', "big"): 
            pass  
        
        elif data[0] == int.from_bytes(b'\x0A', "big"): 
            pass
        elif data[0] == int.from_bytes(b'\x0B', "big"): 
            if data[4] == 1:
                return "Fire", "True"
            else:
                return "Err", "Fire False"
            
        elif data[0] == int.from_bytes(b'\x0C', "big"): 
            if data[4] == 0:
                return "Fire", "False"
            else:
                return "Err", "Fire True"
        
        elif data[0] == int.from_bytes(b'\xAA', "big"): 
            self.CurrentStep = 0
            #self.ScanMode = False
            self.Homed = True
            return "Homed","-" 
        
        elif data[0] == int.from_bytes(b'\xEE', "big"):
            errorMSG = ""
            
            Err_bit = f'{data[1]:08b}'    #0000 1= (MS2 Limit) 1 =(ECE)  1 = (PLC) 1= (MCE)
            Err_bitR = f'{data[2]:08b}'    #00000  1 = (PLC) 1 = (MCE)
            #print(str(Err_bit[7]))
            #print(str(Err_bitR))

            if int(Err_bit[7]) == 1:
                self.MCError = True
                errorMSG+="MCError-"
                self.MCError = True
                self.Error = True
            else:
                self.MCError = False
                
            if int(Err_bit[6]) == 1:
                self.PLCError = True
                self.Error = True
                errorMSG+="PLCError-"
            else:
                self.PLCError = False
                
            if int(Err_bit[5]) == 1:
                self.ECError = True
                self.Error = True
                errorMSG+="ECError-"
            else:
                self.ECError = False
                
            if int(Err_bit[4]) == 1:
                self.MS2_Limit = True
                self.Error = True
                errorMSG+="MS2_Limit-"
            else:
                self.MS2_Limit = False
                
            if int(Err_bitR[7]) == 1:
                self.EmergencyStop = True                
                errorMSG+="Emergendy Stop-"
            else:
                self.EmergencyStop = False 
                
            if int(Err_bitR[6]) == 1:
                self.DistanceProtection = True
                errorMSG+="DistanceProtection"
            else:
                self.DistanceProtection = False

            
            if not (self.MCError) and not(self.PLCError) and not(self.ECError) and not(self.MS2_Limit):
                self.Error = False
                self.log("NoError")
                return "Error","NoError"
            
            print("Error_PoseCtrl")
            self.log(errorMSG)
            self.askErrCode(data) 
            
            return "Error","PoseDevice"
              
        elif data[0] == int.from_bytes(b'\x0E', "big"): 

            if data[1] == 1:
                self.MCE_code = str(data[2]) + str(data[3])
                msg_return = "MCErrCode:" + str(data[2]) + str(data[3])
                return msg_return, "-"

            elif data[1] == 2:
                self.PLCE_code = str(data[2]) + str(data[3])
                msg_return = "PLErrCode:" + str(data[2]) + str(data[3])
                return msg_return, "-" 
                
            elif data[1] == 3:
                self.ECE_code = str(data[2]) + str(data[3])
                msg_return = "ECErrCode:" + str(data[2]) + str(data[3])
                return msg_return, "-" 
    
    
    ### dummy function, real one should measure temperature through the sensor and return that
    def get_temp(self):
        self.temp -= 1
        return self.temp

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
