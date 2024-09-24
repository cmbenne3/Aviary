import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.landing_gear import (
    AltLandingGearMass, LandingGearMass, MainGearLength, NoseGearLength)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class LandingGearMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "landing_gear",
            LandingGearMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                        Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                        Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                        Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                        Aircraft.Design.TOUCHDOWN_MASS],
            output_keys=[Aircraft.LandingGear.MAIN_GEAR_MASS,
                         Aircraft.LandingGear.NOSE_GEAR_MASS],
            version=Version.TRANSPORT,
            atol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class LandingGearMassTest2(unittest.TestCase):

    def setUp(self):
        import aviary.subsystems.mass.flops_based.landing_gear as gear
        gear.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.landing_gear as gear
        gear.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "landing_gear",
            LandingGearMass(aviary_options=get_flops_inputs("N3CC")),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 100.0, 'inch')
        prob.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 75.0, 'inch')
        prob.set_val(Aircraft.Design.TOUCHDOWN_MASS, 100000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class AltLandingGearMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "landing_gear_alt",
            AltLandingGearMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                        Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                        Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                        Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                        Mission.Design.GROSS_MASS],
            output_keys=[Aircraft.LandingGear.MAIN_GEAR_MASS,
                         Aircraft.LandingGear.NOSE_GEAR_MASS],
            version=Version.ALTERNATE,
            atol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltLandingGearMassTest2(unittest.TestCase):

    def setUp(self):
        import aviary.subsystems.mass.flops_based.landing_gear as gear
        gear.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.landing_gear as gear
        gear.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "landing_gear_alt",
            AltLandingGearMass(aviary_options=get_flops_inputs("N3CC")),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 100.0, 'inch')
        prob.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 75.0, 'inch')
        prob.set_val(Mission.Design.GROSS_MASS, 100000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class LandingGearLengthTest(unittest.TestCase):
    """
    This component is unrepresented in our test data.
    """

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only='N3CC'),
                          name_func=print_case)
    def test_derivs(self, case_name):
        prob = self.prob
        model = prob.model
        flops_inputs = get_flops_inputs(case_name)

        model.add_subsystem(
            'main', MainGearLength(aviary_options=flops_inputs), promotes=['*'])
        model.add_subsystem(
            'nose', NoseGearLength(aviary_options=flops_inputs), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.LENGTH,
                        Aircraft.Fuselage.MAX_WIDTH,
                        Aircraft.Nacelle.AVG_DIAMETER,
                        Aircraft.Engine.WING_LOCATIONS,
                        Aircraft.Wing.DIHEDRAL,
                        Aircraft.Wing.SPAN],
            output_keys=[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                         Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH],
            version=Version.ALTERNATE,
            atol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
