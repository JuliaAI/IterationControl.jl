m = SquareRooter(4)
report = IC.train!(m, Train(2),  NotANumber(), NumberLimit(3); verbosity=0);
@test report[1] == (Train(2), NamedTuple())
@test report[2] == (NotANumber(), (done=false, log=""))
report[3] == (NumberLimit(3),
                    (done=true,
                     log="Early stop triggered by NumberLimit(3) "*
                     "stopping criterion. "))

m = SquareRooter(4)
@test_logs((:info, r"Early stop triggered by Num"),
           IC.train!(m, Train(2),  NotANumber(), NumberLimit(3)));
@test_logs((:info, r"Using these controls"),
           (:info, r"Training model for 2 iterations"),
           (:info, r"Training model for 2 iterations"),
           (:info, r"Training model for 2 iterations"),
           (:info, r"Early stop triggered by NumberLimit"),
           IC.train!(m, Train(2),  NotANumber(), NumberLimit(3); verbosity=2));

