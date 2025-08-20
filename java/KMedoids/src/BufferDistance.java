import elki.database.ids.DBIDRange;
import elki.distance.AbstractDBIDRangeDistance;

import java.nio.FloatBuffer;

public class BufferDistance extends AbstractDBIDRangeDistance {
	private final FloatBuffer fb;
	private final int n;
	
	public BufferDistance(FloatBuffer fb, int n) {
		this.fb = fb;
		this.n = n;
	}
	
	@Override
	public double distance(int i, int j) {
		return fb.get(i * n + j);
	}
	
	@Override
	public void checkRange(DBIDRange dbidRange) {
	
	}
	
	@Override
	public boolean isSymmetric() {
		return false;
	}
}
