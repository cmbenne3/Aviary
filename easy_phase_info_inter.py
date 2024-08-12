phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": True},
    "climb_1": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 3,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.3, "unitless"),
            "final_mach": (0.7, "unitless"),
            "mach_bounds": ((0.27999999999999997, 0.72), "unitless"),
            "initial_altitude": (0.0, "ft"),
            "final_altitude": (24500.0, "ft"),
            "altitude_bounds": ((0.0, 25000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": True,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 0.0), "min"),
            "duration_bounds": ((35.0, 105.0), "min"),
        },
        "initial_guesses": {"time": ([0.0, 70.0], "min")},
    },
    "climb_2": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 3,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.7, "unitless"),
            "final_mach": (0.7, "unitless"),
            "mach_bounds": ((0.6799999999999999, 0.72), "unitless"),
            "initial_altitude": (24500.0, "ft"),
            "final_altitude": (25000.0, "ft"),
            "altitude_bounds": ((24000.0, 25500.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((35.0, 105.0), "min"),
            "duration_bounds": ((245.0, 735.0), "min"),
        },
        "initial_guesses": {"time": ([70.0, 490.0], "min")},
    },
    "descent_1": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 3,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.7, "unitless"),
            "final_mach": (0.3, "unitless"),
            "mach_bounds": ((0.27999999999999997, 0.72), "unitless"),
            "initial_altitude": (25000.0, "ft"),
            "final_altitude": (0.0, "ft"),
            "altitude_bounds": ((0.0, 25500.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((280.0, 840.0), "min"),
            "duration_bounds": ((15.0, 45.0), "min"),
        },
        "initial_guesses": {"time": ([560.0, 30.0], "min")},
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": True,
        "target_range": (4367.14, "nmi"),
    },
}