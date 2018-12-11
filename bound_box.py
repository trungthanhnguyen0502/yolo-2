class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c, self.probs = 0., 0., 0., 0., 0., 0.
    
    def iou(self, box):
        if( self.h * self.w > box.h* box.w):
            return 1
        elif(box.probs >= 0.5 and self.probs - box.probs <= 0.1 ):
            return -1
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union

    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3