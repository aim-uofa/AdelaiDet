import numpy as np

from detectron2.utils.visualizer import Visualizer


class TextVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        beziers = predictions.beziers.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs

        self.overlay_instances(beziers, recs, scores)

        return self.output

    def _bezier_to_poly(self, bezier):
        # bezier to polygon
        u = np.linspace(0, 1, 20)
        bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
        points = np.outer((1 - u) ** 3, bezier[:, 0]) \
            + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
            + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
            + np.outer(u ** 3, bezier[:, 3])
        points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)

        return points

    def _decode_recognition(self, rec):
        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

        s = ''
        for c in rec:
            c = int(c)
            if c < 95:
                s += CTLABELS[c]
            elif c == 95:
                s += u'口'
        return s

    def _ctc_decode_recognition(self, rec):
        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

        # ctc decoding
        last_char = False
        s = ''
        for c in rec:
            c = int(c)
            if c < 95:
                if last_char != c:
                    s += CTLABELS[c]
                    last_char = c
            elif c == 95:
                s += u'口'
            else:
                last_char = False
        return s

    def overlay_instances(self, beziers, recs, scores, alpha=0.5):
        color = (0.1, 0.2, 0.5)

        for bezier, rec, score in zip(beziers, recs, scores):
            polygon = self._bezier_to_poly(bezier)
            self.draw_polygon(polygon, color, alpha=alpha)

            # draw text in the top left corner
            text = self._decode_recognition(rec)
            text = "{:.3f}: {}".format(score, text)
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            text_pos = polygon[0]
            horiz_align = "left"
            font_size = self._default_font_size

            self.draw_text(
                text,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
            )