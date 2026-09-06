import math


class CentroidTracker:

    def __init__(
        self,
        max_disappeared=45,
        max_distance=250,
        min_iou=0.05
    ):
        self.next_id = 0

        self.objects = {}
        self.disappeared = {}

        self.previous_centers = {}
        self.velocities = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou = min_iou

    # ========================================================
    # CENTER
    # ========================================================

    def _center(self, box):
        x1, y1, x2, y2 = box

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    # ========================================================
    # DISTANCE
    # ========================================================

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # ========================================================
    # IOU
    # ========================================================

    def _iou(self, box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0,
            ix2 - ix1
        )

        ih = max(
            0,
            iy2 - iy1
        )

        intersection = iw * ih

        area_a = max(
            0,
            ax2 - ax1
        ) * max(
            0,
            ay2 - ay1
        )

        area_b = max(
            0,
            bx2 - bx1
        ) * max(
            0,
            by2 - by1
        )

        union = (
            area_a +
            area_b -
            intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ========================================================
    # BOX SIZE
    # ========================================================

    def _box_area(self, box):

        x1, y1, x2, y2 = box

        return max(
            0,
            x2 - x1
        ) * max(
            0,
            y2 - y1
        )

    # ========================================================
    # BOX CENTER DISTANCE NORMALIZATION
    # ========================================================

    def _normalized_center_distance(
        self,
        box_a,
        box_b
    ):

        center_a = self._center(box_a)
        center_b = self._center(box_b)

        distance = self._distance(
            center_a,
            center_b
        )

        # Normalize by average box diagonal.
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        diag_a = math.sqrt(
            (ax2 - ax1) ** 2 +
            (ay2 - ay1) ** 2
        )

        diag_b = math.sqrt(
            (bx2 - bx1) ** 2 +
            (by2 - by1) ** 2
        )

        avg_diag = max(
            1.0,
            (diag_a + diag_b) / 2
        )

        return distance / avg_diag

    # ========================================================
    # REGISTER
    # ========================================================

    def register(self, box):

        object_id = self.next_id

        self.next_id += 1

        center = self._center(box)

        self.objects[object_id] = {
            "box": box,
            "center": center
        }

        self.previous_centers[object_id] = center

        self.velocities[object_id] = (
            0.0,
            0.0
        )

        self.disappeared[object_id] = 0

    # ========================================================
    # DEREGISTER
    # ========================================================

    def deregister(self, object_id):

        self.objects.pop(
            object_id,
            None
        )

        self.disappeared.pop(
            object_id,
            None
        )

        self.previous_centers.pop(
            object_id,
            None
        )

        self.velocities.pop(
            object_id,
            None
        )

    # ========================================================
    # PREDICT POSITION
    # ========================================================

    def _predict_position(self, object_id):

        current = self.objects[
            object_id
        ]["center"]

        vx, vy = self.velocities.get(
            object_id,
            (0.0, 0.0)
        )

        # Limit prediction so a noisy velocity does not
        # move the predicted point too far.
        vx = max(
            -100,
            min(100, vx)
        )

        vy = max(
            -100,
            min(100, vy)
        )

        return (
            int(current[0] + vx),
            int(current[1] + vy)
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, boxes):

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if len(boxes) == 0:

            for object_id in list(
                self.disappeared.keys()
            ):

                self.disappeared[
                    object_id
                ] += 1

                if (
                    self.disappeared[
                        object_id
                    ] > self.max_disappeared
                ):

                    self.deregister(
                        object_id
                    )

            return self.objects

        # ----------------------------------------------------
        # First detection
        # ----------------------------------------------------

        if len(self.objects) == 0:

            for box in boxes:
                self.register(box)

            return self.objects

        object_ids = list(
            self.objects.keys()
        )

        # ----------------------------------------------------
        # Candidate matches
        # ----------------------------------------------------

        candidates = []

        for object_id in object_ids:

            old_box = self.objects[
                object_id
            ]["box"]

            predicted_center = (
                self._predict_position(
                    object_id
                )
            )

            for input_index, new_box in enumerate(
                boxes
            ):

                new_center = self._center(
                    new_box
                )

                distance = self._distance(
                    predicted_center,
                    new_center
                )

                iou = self._iou(
                    old_box,
                    new_box
                )

                normalized_distance = (
                    self._normalized_center_distance(
                        old_box,
                        new_box
                    )
                )

                # ------------------------------------------------
                # Match rules
                #
                # Strong IoU OR reasonable distance.
                # This prevents a tiny unrelated detection
                # from stealing an existing ID.
                # ------------------------------------------------

                valid_match = (
                    iou >= self.min_iou
                    or
                    distance <= self.max_distance
                )

                if not valid_match:
                    continue

                # ------------------------------------------------
                # Box-size similarity
                # ------------------------------------------------

                old_area = max(
                    1,
                    self._box_area(old_box)
                )

                new_area = max(
                    1,
                    self._box_area(new_box)
                )

                area_ratio = (
                    min(old_area, new_area)
                    /
                    max(old_area, new_area)
                )

                # ------------------------------------------------
                # Matching score
                #
                # Higher IoU = better
                # Smaller distance = better
                # Similar box size = better
                # ------------------------------------------------

                score = (
                    (1.0 - iou) * 45.0
                    +
                    normalized_distance * 35.0
                    +
                    (1.0 - area_ratio) * 20.0
                )

                candidates.append(
                    (
                        score,
                        distance,
                        iou,
                        area_ratio,
                        object_id,
                        input_index
                    )
                )

        # Best candidates first
        candidates.sort(
            key=lambda x: x[0]
        )

        used_objects = set()
        used_inputs = set()

        # ----------------------------------------------------
        # Assign matches
        # ----------------------------------------------------

        for (
            score,
            distance,
            iou,
            area_ratio,
            object_id,
            input_index
        ) in candidates:

            if object_id in used_objects:
                continue

            if input_index in used_inputs:
                continue

            # ------------------------------------------------
            # Extra protection:
            # if there is almost no overlap AND the box
            # changed enormously, don't reuse the ID.
            # ------------------------------------------------

            if (
                iou < 0.01
                and
                area_ratio < 0.20
                and
                distance > self.max_distance * 0.50
            ):
                continue

            old_center = self.objects[
                object_id
            ]["center"]

            new_box = boxes[
                input_index
            ]

            new_center = self._center(
                new_box
            )

            # ------------------------------------------------
            # Velocity
            # ------------------------------------------------

            dx = (
                new_center[0]
                - old_center[0]
            )

            dy = (
                new_center[1]
                - old_center[1]
            )

            old_vx, old_vy = (
                self.velocities.get(
                    object_id,
                    (0.0, 0.0)
                )
            )

            # Smooth velocity heavily to reduce
            # detector jitter caused by hand movement.
            alpha = 0.25

            new_vx = (
                alpha * dx
                +
                (1.0 - alpha) * old_vx
            )

            new_vy = (
                alpha * dy
                +
                (1.0 - alpha) * old_vy
            )

            # Prevent velocity explosion.
            new_vx = max(
                -80,
                min(80, new_vx)
            )

            new_vy = max(
                -80,
                min(80, new_vy)
            )

            self.velocities[
                object_id
            ] = (
                new_vx,
                new_vy
            )

            # ------------------------------------------------
            # Save previous center
            # ------------------------------------------------

            self.previous_centers[
                object_id
            ] = old_center

            # ------------------------------------------------
            # Update object
            # ------------------------------------------------

            self.objects[
                object_id
            ] = {
                "box": new_box,
                "center": new_center
            }

            self.disappeared[
                object_id
            ] = 0

            used_objects.add(
                object_id
            )

            used_inputs.add(
                input_index
            )

        # ----------------------------------------------------
        # Unmatched old tracks
        # ----------------------------------------------------

        for object_id in object_ids:

            if object_id not in used_objects:

                self.disappeared[
                    object_id
                ] += 1

                # Keep track alive for temporary
                # detector failures.
                if (
                    self.disappeared[
                        object_id
                    ] > self.max_disappeared
                ):

                    self.deregister(
                        object_id
                    )

        # ----------------------------------------------------
        # New detections
        # ----------------------------------------------------

        for input_index, box in enumerate(
            boxes
        ):

            if input_index not in used_inputs:

                self.register(
                    box
                )

        return self.objects

    # ========================================================
    # MOTION
    # ========================================================

    def get_motion(self, object_id):

        if object_id not in self.objects:

            return (
                0.0,
                "STATIONARY"
            )

        current = self.objects[
            object_id
        ]["center"]

        previous = self.previous_centers.get(
            object_id,
            current
        )

        dx = (
            current[0]
            - previous[0]
        )

        dy = (
            current[1]
            - previous[1]
        )

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        # Ignore detector jitter
        if distance < 3:

            return (
                0.0,
                "STATIONARY"
            )

        # Direction
        if abs(dx) > abs(dy):

            direction = (
                "RIGHT"
                if dx > 0
                else "LEFT"
            )

        else:

            direction = (
                "DOWN"
                if dy > 0
                else "UP"
            )

        return (
            round(distance, 2),
            direction
        )