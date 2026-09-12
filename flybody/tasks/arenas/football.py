"""Football (soccer) arena for fly-vs-fly goal-scoring task."""

from dm_control import composer


class FootballArena(composer.Arena):
    """Flat field with a free ball and a goal at each end (x-axis).

    Field is centered at origin. Goals face inward at
    x = +/- field_length / 2. Ball starts at center.
    Units are cm (same as flybody, fly body ~0.25cm).
    """

    def _build(self,
               field_length: float = 4.0,
               field_width: float = 3.0,
               ball_radius: float = 0.15,
               ball_density: float = 0.02,
               goal_width: float = 1.0,
               goal_height: float = 0.6,
               goal_depth: float = 0.3,
               name: str = 'football'):
        super()._build(name=name)
        self._field_length = field_length
        self._field_width = field_width
        self._ball_radius = ball_radius
        self._goal_width = goal_width
        self._goal_height = goal_height

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[.4, .4, .4],
            diffuse=[.8, .8, .8],
            specular=[.1, .1, .1],
        )

        # Ground texture / material (green pitch with checker).
        self._ground_texture = self._mjcf_root.asset.add(
            'texture',
            rgb1=[.15, .45, .2],
            rgb2=[.1, .35, .15],
            type='2d',
            builtin='checker',
            name='football_pitch',
            width=200,
            height=200,
        )
        self._ground_material = self._mjcf_root.asset.add(
            'material',
            name='football_pitch',
            texrepeat=[4, 3],
            texuniform=False,
            reflectance=0.1,
            texture=self._ground_texture,
        )
        # Ball material (white) and goal material.
        self._ball_material = self._mjcf_root.asset.add(
            'material', name='football_ball', rgba=[1, 1, 1, 1])
        self._goal_material = self._mjcf_root.asset.add(
            'material', name='football_goal', rgba=[1, 1, 1, 1])

        # Ground plane.
        self._ground_geom = self._mjcf_root.worldbody.add(
            'geom',
            type='plane',
            size=(field_length, field_width, 0.1),
            material=self._ground_material,
        )

        # Center circle visual (thin ring drawn as flat cylinder, no contacts).
        self._mjcf_root.worldbody.add(
            'geom',
            type='cylinder',
            pos=(0, 0, 0.001),
            size=(0.5, 0.005),
            rgba=[1, 1, 1, 0.35],
            contype=0,
            conaffinity=0,
        )

        # Free ball at center.
        ball_body = self._mjcf_root.worldbody.add(
            'body', name='football', pos=(0, 0, ball_radius + 0.01))
        self._ball_joint = ball_body.add('joint', name='football',
                                         type='free')
        self._ball_geom = ball_body.add(
            'geom',
            name='football',
            type='sphere',
            size=(ball_radius,),
            material=self._ball_material,
            density=ball_density,
            friction=(0.5, 0.1, 0.1),
        )
        self._ball_body = ball_body

        # Fixed overview camera that tracks the ball, for recording videos.
        # Render it with: physics.render(camera_id='overview').
        self._mjcf_root.worldbody.add(
            'camera',
            name='overview',
            mode='targetbody',
            target='football',
            pos=(0, -field_width / 2.0 - 3.0, 2.5),
        )

        # Goals at both ends, opening faces the field center.
        post_radius = 0.03
        for side, sign in (('east', 1.0), ('west', -1.0)):
            gx = sign * field_length / 2.0
            # Two vertical posts.
            for postsuffix, py in (('left', -goal_width / 2.0),
                                   ('right', goal_width / 2.0)):
                self._mjcf_root.worldbody.add(
                    'geom',
                    name=f'goal_{side}_post_{postsuffix}',
                    type='cylinder',
                    pos=(gx, py, goal_height / 2.0),
                    size=(post_radius, goal_height / 2.0),
                    material=self._goal_material,
                )
            # Crossbar (along y, positioned via fromto only).
            self._mjcf_root.worldbody.add(
                'geom',
                name=f'goal_{side}_bar',
                type='capsule',
                fromto=(gx, -goal_width / 2.0, goal_height,
                        gx, goal_width / 2.0, goal_height),
                size=(post_radius,),
                material=self._goal_material,
            )
            # Back net visual (contact-free thin box behind goal line).
            self._mjcf_root.worldbody.add(
                'geom',
                name=f'goal_{side}_net',
                type='box',
                pos=(gx + sign * goal_depth / 2.0, 0, goal_height / 2.0),
                size=(goal_depth / 2.0, goal_width / 2.0, goal_height / 2.0),
                rgba=[0.9, 0.9, 0.9, 0.15],
                contype=0,
                conaffinity=0,
            )

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def ball_radius(self):
        return self._ball_radius

    @property
    def field_length(self):
        return self._field_length

    @property
    def field_width(self):
        return self._field_width

    @property
    def goal_width(self):
        return self._goal_width

    @property
    def goal_height(self):
        return self._goal_height

    @property
    def east_goal_x(self):
        return self._field_length / 2.0

    @property
    def west_goal_x(self):
        return -self._field_length / 2.0

    def regenerate(self, random_state):
        pass
