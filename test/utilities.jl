@testset "doc string generation" begin
    IterationControl.@create_docs(SquareRooter,
                              header="header",
                              example="example",
                              body="body")

    paragraphs = split(string(@doc SquareRooter), "\n")
    @test paragraphs[2] == "header"
    @test paragraphs[end-1] == "body"
end

true
