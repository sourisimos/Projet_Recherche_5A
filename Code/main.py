from maillage import MaillageDelaunayMultiDimension

if __name__ == "__main__":
    m = MaillageDelaunayMultiDimension(100, 2, 1)
    x = [0.5,0.5]
    m.evaluate_function_at_point(x)
    m.plot()
