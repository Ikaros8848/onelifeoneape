"""Shape-faithful 4x4, 16-map software model."""
from .scamp_ops import conv2d_ternary, maxpool2d, relu

class ScampCNN:
    def __init__(self, filters=None, fc=None):
        # Accept both the firmware-shaped tensors (16,1,4,4) and the simple
        # nested lists used by the dependency-free implementation.
        if filters is None:
            self.filters = [[[0]*4 for _ in range(4)] for _ in range(16)]
        else:
            self.filters = []
            for f in filters:
                if len(f) == 1 and hasattr(f[0], '__len__'):
                    f = f[0]
                self.filters.append([[int(v) for v in row] for row in f])
        self.fc = None
        if fc is not None:
            # Original header layout is (10,16,16,16); FC consumes a flat
            # 16x16 map per filter after 4x4 pooling.
            self.fc = []
            for row in fc:
                if hasattr(row, 'reshape'):
                    row = row.reshape(-1).tolist()
                elif row and hasattr(row[0], '__len__'):
                    row = [v for plane in row for r in plane for v in r]
                self.fc.append([int(v) for v in row])
    def forward(self, image):
        # Firmware keeps a 64x64 PE plane; emulate its border-preserving
        # placement by padding the 64x64 software image to 67x67 before the
        # valid 4x4 stencil, yielding a 64x64 map.
        h,w=len(image),len(image[0]); padded=[[0.0]*(w+3) for _ in range(h+3)]
        for y in range(h):
            for x in range(w): padded[y+1][x+1]=image[y][x]
        maps=[relu(conv2d_ternary(padded,f)) for f in self.filters]
        pooled=[maxpool2d(m, size=4, stride=4) for m in maps]
        flat=[v for m in pooled for r in m for v in r]
        logits=[sum(flat[i]*row[i] for i in range(min(len(flat),len(row)))) for row in self.fc] if self.fc else [0.0]*10
        return {'conv':maps,'activation':maps,'pool':pooled,'fc':logits,'logits':logits,'pred':max(range(10),key=lambda i:logits[i])}
