"""
Curated list of aviary inputs that are promoted as parameters in the mission.
"""
from aviary.variable_info.variables import Aircraft, Mission


core_mission_inputs = [
    Aircraft.Design.BASE_AREA,
    Aircraft.Design.DRAG_POLAR,
    Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
    Aircraft.Design.LIFT_POLAR,
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
    Aircraft.Engine.SCALE_FACTOR,
    Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
    Aircraft.Fuselage.CROSS_SECTION,
    Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
    Aircraft.Fuselage.FINENESS,
    Aircraft.Fuselage.LAMINAR_FLOW_LOWER,
    Aircraft.Fuselage.LAMINAR_FLOW_UPPER,
    Aircraft.Fuselage.LENGTH_TO_DIAMETER,
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
    Aircraft.HorizontalTail.FINENESS,
    Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER,
    Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER,
    Aircraft.HorizontalTail.WETTED_AREA,
    Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
    Aircraft.Nacelle.FINENESS,
    Aircraft.Nacelle.LAMINAR_FLOW_LOWER,
    Aircraft.Nacelle.LAMINAR_FLOW_UPPER,
    Aircraft.Nacelle.WETTED_AREA,
    Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
    Aircraft.VerticalTail.FINENESS,
    Aircraft.VerticalTail.LAMINAR_FLOW_LOWER,
    Aircraft.VerticalTail.LAMINAR_FLOW_UPPER,
    Aircraft.VerticalTail.WETTED_AREA,
    Aircraft.Wing.AREA,
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.CHARACTERISTIC_LENGTH,
    Aircraft.Wing.FINENESS,
    Aircraft.Wing.INCIDENCE,
    Aircraft.Wing.LAMINAR_FLOW_LOWER,
    Aircraft.Wing.LAMINAR_FLOW_UPPER,
    Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
    Aircraft.Wing.SPAN_EFFICIENCY_FACTOR,
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD,
    Aircraft.Wing.WETTED_AREA,
    Mission.Design.GROSS_MASS,
    Mission.Design.LIFT_COEFFICIENT,
    Mission.Design.MACH,
    Mission.Takeoff.FINAL_MASS,
]
