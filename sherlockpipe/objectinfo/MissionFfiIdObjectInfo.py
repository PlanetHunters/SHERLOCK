from objectinfo.ObjectInfo import ObjectInfo

class MissionFfiIdObjectInfo(ObjectInfo):
    def __init__(self, mission_id, sectors):
        super().__init__()
        self.id = mission_id
        self.sectors = sectors

    def sherlock_id(self):
        return "FFI_" + self.id + "_" + str(self.sectors)

    def mission_id(self):
        return self.id

